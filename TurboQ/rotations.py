import torch
from math import sqrt

def generate_rotation_matrix(
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate a random orthogonal matrix Π ∈ R^{d×d} via QR decomposition.

    This is the method described in Algorithm 1 of the paper.
    For head_dim=128, this is a 128×128 matrix = 64KB in float32, negligible.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    # Generate on CPU for reproducibility, then move to device
    G = torch.randn(d, d, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)

    # Ensure proper rotation (det = +1) by fixing signs
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)

    return Q.to(device=device, dtype=dtype)


def generateQJLMatrix(
        d: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        seed: int = 310805 
) -> torch.Tensor:
    
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed=seed)
    S = torch.randn(d,d, generator=gen, dtype=torch.float32)
    return S.to(device=device, dtype=dtype)


def generateRademacher(
    dim: int,
    device: torch.device = 'cpu',
    dtype: torch.dtype = torch.float32,
    seed: int = 310805
) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(seed)

    rademacherVector = torch.empty(dim)
    rademacherVector.random_(0,2, generator=rng)
    rademacherVector[rademacherVector == 0] = -1
    return rademacherVector.to(device=device, dtype=dtype)



#1. Rotation
def FWHT(
    key: torch.Tensor,
    inplace: bool = True,
    normalize: bool = False
) -> torch.Tensor:
    data = key if inplace else key.clone()
    
    n = data.size(dim=-1)
    h = 1
    while h < n:
        data_reshaped = data.reshape(*data.shape[:-1], n//(2*h), 2, h)
        x = data_reshaped[..., 0, :].clone()
        y = data_reshaped[..., 1, :]
        data_reshaped[..., 0, :] = x+y
        data_reshaped[..., 1, :] = x-y
        h *= 2
    if normalize:
        print("Gonna normalize")
        data /= sqrt(n)
        # print("Normalized Tensor: ", data)
    # print(data.shape)
    return data

# def rotate_forward(x: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
#     """Apply random rotation: y = x @ Pi^T (equivalent to Pi @ x for each vector)."""
#     return torch.matmul(x, Pi.T)


# def rotate_backward(y: torch.Tensor, Pi: torch.Tensor) -> torch.Tensor:
#     """Apply inverse rotation: x = y @ Pi (equivalent to Pi^T @ y)."""
#     return torch.matmul(y, Pi)

def forward_rotation(
    x: torch.Tensor,
    D1: torch.Tensor,
) ->  torch.Tensor:
    x_ran = torch.mul(D1,x)
    x_rot1 = FWHT(x_ran, normalize=True)
    return x_rot1

def backward_rotation(
    x: torch.Tensor,       
    D1: torch.Tensor,
) ->  torch.Tensor:
    x_rot1 = FWHT(x, normalize=True)
    x_ran = torch.mul(D1,x_rot1)
    return x_ran

# def forward_rotation(
#     x: torch.Tensor,
#     D1: torch.Tensor,
#     D2: torch.Tensor
# ) ->  torch.Tensor:
#     x_ran = torch.mul(D1,x)
#     x_rot1 = FWHT(x_ran, normalize=True)
#     x_ran2 = torch.mul(D2,x_rot1)
#     x_rot2 = FWHT(x_ran2, normalize=True)
#     return x_rot2
#     # return x_rot1

# def backward_rotation(
#     x: torch.Tensor,       
#     D1: torch.Tensor,
#     D2: torch.Tensor,
# ) ->  torch.Tensor:
#     x_rot2 = FWHT(x, normalize=True)
#     x_ran2 = torch.mul(D2,x_rot2)
#     x_rot1 = FWHT(x, normalize=True)
#     x_ran = torch.mul(D1,x_rot1)
#     return x_ran