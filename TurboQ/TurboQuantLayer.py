import torch
from transformers.cache_utils import (
    DynamicSlidingWindowLayer,
    LinearAttentionLayer,
    LinearAttentionAndFullAttentionLayer,
    CacheLayerMixin,
    DynamicCache,
    Cache
)
from collections.abc import Iterable
from transformers.configuration_utils import PreTrainedConfig
from .TurboQuantOperations import (
    TurboQuantMSE,
    TurboQuantResidual,
    TurboQJLPack,
    TurboQMSEPack
)

class TurboQuantCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as a list of `CacheLayer`, one for each layer. The expected shape for each tensor
    in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
    If a config is passed, it will additionally check for sliding or hybrid cache structure, greatly reducing the
    memory requirement of the cached tensors to `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        ddp_cache_data (`Iterable[tuple[torch.Tensor, torch.Tensor]]`, *optional*):
            It was originally added for compatibility with `torch.distributed` (DDP). In a nutshell, it is
            `map(gather_map, zip(*caches))`, i.e. each item in the iterable contains the key and value states
            for a layer gathered across replicas by torch.distributed (shape=[global batch size, num_heads, seq_len, head_dim]).
            Note: it needs to be the 1st arg as well to work correctly
        config (`PreTrainedConfig`, *optional*):
            The config of the model for which this Cache will be used. If passed, it will be used to check for sliding
            or hybrid layer structure, greatly reducing the memory requirement of the cached tensors to
            `[batch_size, num_heads, min(seq_len, sliding_window), head_dim]`.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `False`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> past_key_values = DynamicCache(config=model.config)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    ```
    """

    def __init__(
        self,
        device: torch.Device,
        num_key_heads: int,
        ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
        config: PreTrainedConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
        quant_size: int = 4,
        dim: int = None,
        seed: int = 310805,
    ):
        self.device = device
        layers = []
        self.tq_qjl = TurboQuantResidual(device=device, bits=quant_size, dim=dim, seed=seed)
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
            print("Appending Dynamic Layer for config")
            layers = [TurboQuantLayer(tq_qjl=self.tq_qjl) * len(layer_types)]

        if len(layers) == 0:
            super().__init__(
                layer_class_to_replicate=TurboQuantLayer,
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
        tq_qjl: TurboQuantResidual
    ):
        self.tq_qjl = tq_qjl
    
    def lazy_initialization(self, key_states, value_states) -> None:
            self.dtype, self.device = key_states.dtype, key_states.device
            self.key_cache = TurboQJLPack(
                mse_indices=torch.tensor([], dtype=self.dtype, device=self.device),
                mse_norms=torch.tensor([], dtype=self.dtype, device=self.device),
                residual_pack=torch.tensor([], dtype=self.dtype, device=self.device),
                residual_norms=torch.tensor([], dtype=self.dtype, device=self.device),
            )
            self.values=torch.tensor([], dtype=self.dtype, device=self.device)
            self.is_iniitialized = True

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
        # print(f"Shape of keys sent for caching: {key_states.shape}, values: {value_states.shape}")

        # key_states = TurboQuantOps.FWHT(key_states)
        self._store(key_states=key_states)
        self.keys = self.tq_qjl.dequantize(self.key_cache)
        self.values = torch.cat([self.values, value_states], dim=-2)

        return self.keys, self.values           

    def _store(self, key_states):
        key_quant = self.tq_qjl.quantize(key_states)
        self.key_cache = self._append(key_quant)

    def _append(self, data: TurboQJLPack):
        return TurboQJLPack(
            mse_indices=torch.cat((self.key_cache.mse_indices,data.mse_indices), dim=-2),
            mse_norms=torch.cat((self.key_cache.mse_norms,data.mse_norms), dim=-2),
            residual_pack=torch.cat((self.key_cache.residual_pack,data.residual_pack), dim=-2),
            residual_norms=torch.cat((self.key_cache.residual_norms,data.residual_norms), dim=-2)
        )

        

