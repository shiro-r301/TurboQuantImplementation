import math
import torch.nn.functional as F
import torch
from math import sqrt, pi
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
import os
from TurboQuantOperations import (
    TurboQuantMSE,
    TurboQuantResidual
)

def pipeline(vector: torch.Tensor, device):
    #1. conversion to unit hypersphere X->X_n
        #rotation X_n -> X_rot
    vector = vector.to(device=device, dtype=torch.float32)
    TQMse = TurboQuantMSE(
        dim=vector.shape[-1],
        bits=3,
        device=torch.device('cuda'),
        dtype=torch.float32
    )

    TQProd = TurboQuantResidual(
        device=torch.device('cuda'),
        dim=128,
        dtype=torch.float32
    )
    #2. quantization X_rot -> X_quant
        #indexing : X_quant contains indices that map to respective quantization level
        #bitpacking : Fit x bit values into 8bits
    rng = torch.Generator(device='cpu')
    rng.manual_seed(210777)
    x_quant = TQMse.quantize(vector)
    x_mse = TQMse.dequantize(x_quant)
    #3. residual calc -> X_rot - Dequant(X_quant) -> e
    residual = vector - x_mse
    # print("Actual Residual: ", residual)
    #4. project residual: sign(S.e)
    r_qjl, e_norm = TQProd.quantize(residual)
    #5. store signs : bit-packing 8 values per byte
    x_qjl = TQProd.dequantize(r_qjl, e_norm=e_norm)
    x_f = x_mse + x_qjl
    errors_yours = run_bias_test(vector, x_f, n_samples=1000000)
    print("Yours - mean signed error:", errors_yours.mean().item())
    print("Yours - std:", errors_yours.std().item())

    #inference: 
 
       #QK = Q.Dequant(K_quant) + QS.signs(S.e)



def run_bias_test(vector, x_f, n_samples=1_000_000, seed=210777, device='cuda', batch_size=100_000):
    d = vector.shape[-1]
    vector = vector.to(device=device)
    x_f = x_f.to(device=device)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    errors = torch.empty(n_samples, device=device)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        b = end - start

        # (b, d) batch of random query vectors, generated directly on GPU
        Y = torch.randn(b, d, generator=rng, device=device, dtype=torch.float32)

        true_ip = Y @ vector          # (b,)
        approx_ip = Y @ x_f           # (b,)

        errors[start:end] = approx_ip - true_ip

    return errors


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

pipeline(a, device=torch.device('cuda'))
