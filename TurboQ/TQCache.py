import torch
from transformers.cache_utils import (
    DynamicSlidingWindowLayer,
    LinearAttentionLayer,
    LinearAttentionAndFullAttentionLayer,
    CacheLayerMixin,
    DynamicCache,
    Cache
)
from functools import partial
from collections.abc import Iterable
from transformers.configuration_utils import PreTrainedConfig
from .TurboQuantOperations import (
    TurboQuantMSE,
    TurboQuantResidual,
    TurboQJLPack,
    NoOpQuantizer
)

class TurboQuantCache(Cache):
    """
    kquant_size: by default 4, change for respective level of quantization.
    """
    def __init__(
        self,
        device: torch.device,
        dim: int,
        num_key_heads: int,
        value_quantizer: type[TurboQuantMSE] | type[TurboQuantResidual] | None = None,
        key_quantizer: type[TurboQuantMSE] | type[TurboQuantResidual] | None = None,
        ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
        config: PreTrainedConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
        kquant_size: int = 8,
        vquant_size: int = 16, 
        seed: int = 310805,
    ):
        
        self.dim=dim
        self.device = device
        layers = []

        # print(dim)
        self.tq_key = NoOpQuantizer()
        if key_quantizer is not None:
            print(f"Initializing Key Quantizer:{key_quantizer}")
            self.tq_key = key_quantizer(device=device, bits=kquant_size, dim=dim, seed=seed)

        self.tq_value = NoOpQuantizer()
        if value_quantizer is not None:
            print(f"Initializing Value Quantizer:{value_quantizer}")
            self.tq_value = value_quantizer(device=device, bits=vquant_size, dim=dim, seed=seed)
        
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
                decoder_config, "attention_chunk_size", None
            )
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                layer_types = []
                for _ in range(decoder_config.num_hidden_layers):
                    if sliding_window is not None:
                        layer_types.append("sliding_attention")
                    else:
                        layer_types.append("full_attention")
            # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
            print(f"Applying TurboQuantLayers: ")
            layers = [TurboQuantLayer(tq_key=self.tq_key, tq_value=self.tq_value) for _ in range(len(layer_types))]

        if len(layers) == 0:
            print("Performing Layer Replication")
            super().__init__(
                layer_class_to_replicate=partial(TurboQuantLayer, self.tq_key, self.tq_value),
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding,
            )
        else:
            super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
                    
    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, getattr(layer, "_sliding_window_tensor", None)




class TurboQuantLayer(CacheLayerMixin):
    def __init__(
        self,
        tq_key: TurboQuantResidual | TurboQuantMSE | NoOpQuantizer,
        tq_value: TurboQuantResidual | TurboQuantMSE | NoOpQuantizer,
    ):
        super().__init__()
        self.tq_key = tq_key
        self.tq_value = tq_value
        self.perform_value_quant = True

    def lazy_initialization(self, key_states, value_states) -> None: #abstract class from CacheLayerMixin
        self.values = self.tq_value.initialize_states(value_states)
        self.keys = self.tq_key.initialize_states(key_states)
        self.is_initialized = True

    def get_mask_sizes(self, query_length): #abstract class from CacheLayerMixin
        kv_offset = 0
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self): #abstract class from CacheLayerMixin
        if isinstance(self.tq_key, NoOpQuantizer):
            if not self.is_initialized or self.keys.numel() == 0:
                return 0
            return self.keys.shape[-2]    
        
        if not self.is_initialized or self.keys.mse_indices.numel() == 0:
            return 0
        return self.keys.mse_indices.shape[-2]
    
    def get_max_cache_shape(self) -> int: #abstract class from CacheLayerMixin
        """Returns the maximum sequence length of the cache object. My implementation is built on DynamicLayer. 
        DynamicLayer does not have a maximum length."""
        return -1
    
    def update(
        self, 
        key_states: torch.tensor, 
        value_states: torch.tensor,
        *args, 
        **kwargs
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """

        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        #storage
        self._store_keys(key_states=key_states.to(dtype=torch.float32)) #bf16 dont support right shifts
        dequant_keys = self.tq_key.dequantize(self.keys)

        # print(f"Values recieved: {value_states.shape}")
        self._store_values(value_states=value_states.to(dtype=torch.float32))
        dequant_values = self.tq_value.dequantize(self.values)
        # print(f"Shape of Value after caching: {self.values.shape if isinstance(self.tq_value, NoOpQuantizer) else self.values.mse_indices.shape}, type: {key_states.dtype}. Dequant Size: {dequant_values.shape}")
        # print(f"Shape of New cache : {self.keys.mse_indices.shape}, values: {value_states.shape}, returned Keys: {dequant_keys.shape}")
        return dequant_keys.to(dtype=key_states.dtype), dequant_values.to(dtype=value_states.dtype)           

    def _store_keys(self, key_states):
        key_quant = self.tq_key.quantize(key_states)
        self.keys = self.tq_key.append(self.keys, key_quant)

    def _store_values(self, value_states):
        value_quant = self.tq_value.quantize(value_states)
        self.values = self.tq_value.append(self.values, value_quant)

if __name__ == '__main__':
    print("yas")

        

