import math
import torch.nn.functional as F
import torch
from math import sqrt, pi
from typing import Sequence, NamedTuple
import matplotlib.pyplot as plt
from scipy.stats import kstest
from codebook import get_codebook_tensors
from rotations import (
    generate_rotation_matrix,
    forward_rotation,
    backward_rotation,
    generateQJLMatrix,
    generateRademacher
)


class TurboQJLPack(NamedTuple):
    """
    This is an immutable form of cache storage for storing per layer information 
    of packed mse_index, their norms. Same goes for Residuals. We only need 
    bits and codebook to resolve the x_mse and only e_sign bits and e_norms 
    (e : residual) to retrieve x_qjl
    """
    mse_indices: torch.Tensor
    mse_norms: torch.Tensor
    residual_pack: torch.Tensor
    residual_norms: torch.Tensor


def bit_packing(indices: torch.Tensor, bits: int) -> torch.Tensor:
    d = indices.shape[-1]
    batches = indices.shape[:-1]
    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return indices.to(dtype=torch.uint8)
    
    extra_idx = (vals_per_byte - (d % vals_per_byte)) % vals_per_byte
    indices = F.pad(indices.to(torch.uint8), pad=(0, extra_idx), value=0)
    reshaped_idx = torch.reshape(indices, [*batches, -1, vals_per_byte])
    shifts = torch.arange(0, vals_per_byte, device=indices.device) * bits
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
        bits = 4
    else:
        return packed_idx.to(dtype=torch.long)
    unpacker = (1 << bits) - 1
    shifts = torch.arange(0, vals_per_byte, device=packed_idx.device) * bits
    unpacked_idx = ((packed_idx.unsqueeze(-1) >> shifts) & unpacker).reshape(*batches, -1).to(dtype=torch.long)
    # print("Unpack: ", unpacked_idx[:d].long())
    return unpacked_idx[:d].long()

class TurboQuantMSE(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        dim: int, 
        bits: int = 3,
        seed: int = 310805,
        dtype: torch.dtype = torch.float32
    ): 
        super().__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.dim = dim
        self.b = bits
        self.dtype = dtype
        self.register_buffer('d1', generateRademacher(dim=self.dim, seed=seed+100, device='cuda', dtype=self.dtype))
        # self.register_buffer('d1', generate_rotation_matrix(d=self.dim, device=self.device, dtype=self.dtype, seed=seed))
        centroids, boundaries = get_codebook_tensors(dim, bits, self.device, dtype)
        self.register_buffer("centroids", centroids)      
        self.register_buffer("boundaries", boundaries)
        self.register_buffer("decision_boundaries", boundaries[1:-1].contiguous())

    def quantize(
        self,
        x: torch.Tensor,
    ):
        #unit hypersphere
        x_norm = x.norm(dim=-1)
        x = x / (x_norm.unsqueeze(-1) + 1e-10)

        #rotation
        x_rot = forward_rotation(x, self.d1)
        #quantize
        x_indices = torch.searchsorted(self.decision_boundaries, x_rot.contiguous())
        # print(f"X Packed: {x_indices}")
        x_packed =  bit_packing(x_indices, self.b)
        return x_packed, x_norm

    
    def dequantize(self, pack_idx: torch.Tensor, x_norm: torch.Tensor):
        x_unpack = bit_unpacking(packed_idx=pack_idx, bits=self.b, d=self.dim)
        x_quant = self.centroids[x_unpack[...,:]]
        x_bar = backward_rotation(x_quant,self.d1)
        return x_bar * x_norm.unsqueeze(-1)

class TurboQuantResidual(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        bits: int = 4,
        dim: int = 128,
        seed: int = 210777,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.register_buffer('S', generateQJLMatrix(d=self.dim, device=self.device, dtype=self.dtype, seed=seed))
        self.tq_mse = TurboQuantMSE(device=device, dim=dim, bits=bits-1, seed=seed+3108, dtype=self.dtype)
        self.cf = sqrt(math.pi/2.0)/dim

    def quantize(
        self,
        x: torch.Tensor, #residual
    ) -> TurboQJLPack:
        if dim is None:
            dim = x.shape[-1]
        x_pack, x_norms = self.tq_mse.quantize(x)
        x_mse = self.tq_mse.dequantize(x_pack, x_norms)

        e = x - x_mse 
        e_norm = e.norm(dim=-1)

        e_proj = torch.matmul(e, self.S)
        e_proj[e_proj >= 0] = 1
        e_proj[e_proj < 0] = 0
        e_qjl = bit_packing(e_proj, bits=1)

        return TurboQJLPack(
            mse_indices=x_pack,
            mse_norms=x_norms,
            residual_pack=e_qjl,
            residual_norms=e_norm            
        )


    def dequantize(self, packedCache: TurboQJLPack) -> torch.Tensor:
        x_mse = self.tq_mse.dequantize(packedCache.mse_indices, packedCache.mse_norms) #get mse from packed x
        unpack_Eqjl = bit_unpacking(packedCache.residual_pack, bits=1, d=self.dim) #get unpacked signed residuals
        unpack_Eqjl = 2*unpack_Eqjl - 1
        #Q.(S.T @ Sign(e,S)) * sqrt(pi/2)/d * |e|
        x_qjl = torch.matmul(unpack_Eqjl.to(dtype=self.dtype), self.S.T) 
        x_qjl = x_qjl * (self.cf * packedCache.residual_norms.unsqueeze(-1))
        return x_qjl + x_mse

    def forward(self, x): #can be used to produce 
        return self.dequantize(self.quantize(x))
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
    # ops = TurboQuantMSE(dim=128, bits=4)
    # idx = ops.quantize(a.reshape(1,-1))
    # # print(idx)
    # e = (a - idx).sum()
    # print(e)
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


