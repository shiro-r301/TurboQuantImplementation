import math
import torch.nn.functional as F
import torch
from math import sqrt
from typing import Sequence
import matplotlib.pyplot as plt
from scipy.stats import kstest
from codebook import get_codebook_tensors
from rotations import (
    forward_rotation,
    backward_rotation,
    generateQJLMatrix,
    generateRademacher
)

def pipeline():
    #1. conversion to unit hypersphere X->X_n
        #rotation X_n -> X_rot

    #2. quantization X_rot -> X_quant
        #indexing : X_quant contains indices that map to respective quantization level
        #bitpacking : Fit x bit values into 8bits
        #storage: indices[X_quant], hypersphere_norm, b_bits
    #3. residual calc -> X_rot - Dequant(X_quant) -> e
    #4. project residual: sign(S.e)
    #5. store signs : bit-packing 8 values per byte

    #inference: 
        #Q = Q.Dequant(K_quant) + QS.signs(S.e)
    pass

class TurboQuantMSE(torch.nn.Module):
    def __init__(
        self,
        dim: int, 
        bits: int = 3,
        device: torch.device = 'cpu',
        seed: int = 310805,
        dtype: torch.dtype = torch.float32
    ): 
        super().__init__()
        self.dim = dim
        self.b = bits
        self.device = device
        self.d1 = generateRademacher(dim=self.dim, seed=seed)
        self.d2 = generateRademacher(dim=self.dim, seed=(seed*90 + 170) % 192310)

        centroids, boundaries = get_codebook_tensors(dim, bits, self.device, dtype)
        self.register_buffer("centroids", centroids)      
        self.register_buffer("boundaries", boundaries)
        self.register_buffer("decision_boundaries", boundaries[1:-1].contiguous())

    def quantize(
        self,
        x: torch.Tensor,
    ):
        #unit hypersphere
        self.x_norm = x.norm(dim=-1)
        x = x / (self.x_norm + 1e-9)

        #rotation
        x_rot = forward_rotation(x, self.d1)

        #quantize
        x_indices = torch.searchsorted(self.decision_boundaries, x_rot.contiguous())
        # print(f"X Packed: {x_indices}")
        return self.dequantize(bit_packing(x_indices, self.b))
    
    def dequantize(self, pack_idx: torch.Tensor):
        x_unpack = bit_unpacking(packed_idx=pack_idx, bits=self.b, d=self.dim)
        x_quant = self.centroids[x_unpack[...,:]]
        x_bar = backward_rotation(x_quant,self.d1)
        return x_bar * self.x_norm.unsqueeze(-1)

def bit_packing(indices: torch.Tensor, bits: int) -> torch.Tensor:
    d = indices.shape[-1]
    batches = indices.shape[:-1]
    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
    else:
        return indices.to(dtype=torch.uint8)
    
    extra_idx = (vals_per_byte - (d % vals_per_byte)) % vals_per_byte
    indices = F.pad(indices.to(torch.uint8), pad=(0, extra_idx), value=0)
    reshaped_idx = torch.reshape(indices, [*batches, -1, vals_per_byte])
    shifts = torch.arange(0, vals_per_byte) * bits
    packed = (reshaped_idx << shifts).sum(dim=-1).reshape(*batches, (d+extra_idx) // vals_per_byte)
    return packed

def bit_unpacking(packed_idx: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    batches = packed_idx.shape[:-1]
    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
    else:
        return packed_idx.to(dtype=torch.long)
    unpacker = (1 << bits) - 1
    shifts = torch.arange(0, vals_per_byte) * bits
    unpacked_idx = ((packed_idx.unsqueeze(-1) >> shifts) & unpacker).reshape(*batches, -1).to(dtype=torch.long)
    return unpacked_idx[:d].long()

class TurboQuantResidual():
    def __init__(
        self,
        dim: int = 128,
    ):
        self.S = generateQJLMatrix(dim)

    def quantize():
        pass

    def dequantize():
        pass


if __name__ == '__main__':
    n = 128

    a = torch.Tensor([ 2.2500e+00,  2.0781e+00, -3.6250e+00, -1.7344e+00,  9.0234e-01,
        -8.5156e-01, -2.4062e+00,  9.5312e-01,  8.9844e-01,  6.0156e-01,
        -4.0312e+00, -5.3711e-02,  2.9688e+00,  7.8125e-03,  3.7812e+00,
        -2.5781e+00,  3.1562e+00,  2.5195e-01,  4.7070e-01,  8.0078e-01,
         2.5781e+00, -7.1875e-01,  1.7676e-01,  4.7656e-01, -2.0938e+00,
        -3.8574e-02,  1.7109e+00, -3.3438e+00, -4.5117e-01, -4.1748e-02,
        -1.1953e+00,  3.0664e-01,  5.5859e-01, -1.8750e-01,  2.3340e-01,
        -1.3281e-01,  4.8047e-01, -9.0625e-01, -3.0000e+00, -8.4375e-01,
         6.1328e-01,  4.3359e-01, -1.7676e-01,  1.4688e+00, -5.0391e-01,
         6.0547e-01, -8.0469e-01,  9.8438e-01, -6.5625e-01, -1.8750e+01,
        -5.1562e+00, -4.2812e+00,  7.3750e+00, -7.3750e+00,  2.2344e+00,
        -1.4438e+01,  1.3250e+01, -2.1750e+01,  5.0000e+00,  8.8125e+00,
        -3.3250e+01,  1.8625e+01,  4.2250e+01,  3.9531e+00, -4.2812e+00,
         2.1250e+00, -4.3750e-01,  2.4531e+00, -2.9688e+00, -2.0625e+00,
         2.2344e+00,  2.2500e+00,  2.5781e+00, -1.1797e+00, -2.7656e+00,
         4.6680e-01,  9.4141e-01, -3.5312e+00, -7.8125e-01, -2.0312e+00,
         1.1172e+00, -7.5391e-01,  2.8125e+00,  1.0156e+00,  6.1875e+00,
         5.2734e-01, -1.6797e+00, -1.0078e+00,  5.9766e-01, -1.3750e+00,
        -6.4453e-01, -5.4375e+00, -1.7656e+00, -4.1406e-01, -2.6562e-01,
        -1.5625e-01,  1.5703e+00,  1.1953e+00,  6.0156e-01,  9.5312e-01,
        -1.1016e+00, -2.3438e-01,  9.5625e+00, -1.1328e+00,  8.0469e-01,
         1.8750e+00,  1.7031e+00, -2.5938e+00, -4.5117e-01,  9.7266e-01,
         2.2969e+00,  3.9375e+00,  1.6406e+00, -3.0625e+01,  1.6484e+00,
        -1.4375e+01,  5.4688e-01,  1.1172e+00,  8.5625e+00,  4.9609e-01,
         1.0562e+01, -1.7875e+01, -2.5250e+01, -5.3500e+01,  2.3875e+01,
         5.5750e+01,  2.4750e+01,  8.6000e+01])
    ops = TurboQuantMSE(dim=128, bits=4)
    idx = ops.quantize(a.reshape(1,-1))
    # print(idx)
    e = (a - idx).sum()
    print(e)
    # plt.bar(torch.arange(n), a.flatten().numpy(), width = 1.0)
    # plt.title('Key Vector distribution')
    # plt.savefig('graphs/original_a.png')
    # a = a.reshape(1,1,1,n)


    # b = torch.empty([1,1,1,n], device='cpu')
    # b = b.random_(0,2)
    # b[b == 0] = -1
    # plt.bar(torch.arange(n), b.flatten().numpy(), width = 1.0)
    # plt.title('Rademacher A distribution')
    # plt.savefig('graphs/rademacher_1.png')

    # c = torch.empty([1,1,1,n], device='cpu')
    # c = c.random_(0,2)
    # c[c == 0] = -1
    # plt.bar(torch.arange(n), c.flatten().numpy(), width = 1.0)
    # plt.title('Rademacher B distribution')
    # plt.savefig('graphs/rademacher_2.png')

    # print(f"B: {b}")
    # print(f"C: {c}")

    # #Transformation 1 Z = HD1x
    # x = a #Dx
    # # x = a
    # print(x.shape)
    # r = ops.FWHT(x, normalize=True) #HDx
    # t1 = plt.bar(torch.arange(n), r.flatten().numpy(), width = 1.0)
    # plt.title('Key Vector Transform 1 distribution')
    # plt.savefig('graphs/transform_1.png')
    # t1.remove()

    # #Transformation 2 Z = HD2HD1x
    # f = c*r
    # f = ops.FWHT(f, normalize=True)
    # t2 = plt.bar(torch.arange(n), f.flatten().numpy(), width = 1.0)
    # plt.title('Key Vector Transform 2 distribution')
    # plt.savefig('graphs/transform_2.png')
    # plt.close()

    # #Statistics T1
    # plt.hist(r.flatten().numpy(), bins=15, density=True)
    # plt.title('Key Vector Transform 1 Hist distribution')
    # plt.savefig('graphs/transform_1_hist.png')
    # plt.close()

    # sigma = torch.linalg.norm(r).item() / math.sqrt(n)
    # result = kstest(r.flatten().numpy(), 'norm', args=(0, sigma))
    # print(f"T1: Statistic: {result.statistic}, P-value: {result.pvalue}")

    # #Statistics T2  
    # r2 = (f.float()**2).sum().item()
    # sigma = math.sqrt(r2 / n)
    # result = kstest(
    #     f.flatten().numpy(),
    #     "norm",
    #     args=(0.0, sigma)
    # )
    # print(f"T2: Statistic: {result.statistic}, P-value: {result.pvalue}")
    # plt.hist(f.flatten().numpy(), bins=15, density=True)
    # plt.title('Key Vector Transform 2 Hist distribution')
    # plt.savefig('graphs/transform_2_hist.png')
    # plt.close()


