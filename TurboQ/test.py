# import math
# import torch.nn.functional as F
# import torch
# from math import sqrt, pi
# from typing import Sequence
# import matplotlib.pyplot as plt
# from scipy.stats import kstest
# from codebook import get_codebook_tensors
# from rotations import (
#     forward_rotation,
#     backward_rotation,
#     generateQJLMatrix,
#     generateRademacher
# )
# import os
# from TurboQuantOperations import (
#     TurboQuantMSE,
#     TurboQuantResidual
# )

# def pipeline(vector: torch.Tensor, device):
#     #1. conversion to unit hypersphere X->X_n
#         #rotation X_n -> X_rot
#     vector = vector.to(device=device, dtype=torch.float32)
#     TQMse = TurboQuantMSE(
#         dim=vector.shape[-1],
#         bits=3,
#         device=torch.device('cuda'),
#         dtype=torch.float32
#     )

#     TQProd = TurboQuantResidual(
#         device=torch.device('cuda'),
#         dim=128,
#         dtype=torch.float32
#     )
#     #2. quantization X_rot -> X_quant
#         #indexing : X_quant contains indices that map to respective quantization level
#         #bitpacking : Fit x bit values into 8bits
#     rng = torch.Generator(device='cpu')
#     rng.manual_seed(210777)
#     x_quant = TQMse.quantize(vector)
#     x_mse = TQMse.dequantize(x_quant)
#     #3. residual calc -> X_rot - Dequant(X_quant) -> e
#     residual = vector - x_mse
#     # print("Actual Residual: ", residual)
#     #4. project residual: sign(S.e)
#     r_qjl, e_norm = TQProd.quantize(residual)
#     #5. store signs : bit-packing 8 values per byte
#     x_qjl = TQProd.dequantize(r_qjl, e_norm=e_norm)
#     x_f = x_mse + x_qjl
#     errors_yours = run_bias_test(vector, x_f, n_samples=1000000)
#     print("Yours - mean signed error:", errors_yours.mean().item())
#     print("Yours - std:", errors_yours.std().item())

#     #inference: 
 
#        #QK = Q.Dequant(K_quant) + QS.signs(S.e)



# def run_bias_test(vector, x_f, n_samples=1_000_000, seed=210777, device='cuda', batch_size=100_000):
#     d = vector.shape[-1]
#     vector = vector.to(device=device)
#     x_f = x_f.to(device=device)

#     rng = torch.Generator(device=device)
#     rng.manual_seed(seed)

#     errors = torch.empty(n_samples, device=device)

#     for start in range(0, n_samples, batch_size):
#         end = min(start + batch_size, n_samples)
#         b = end - start

#         # (b, d) batch of random query vectors, generated directly on GPU
#         Y = torch.randn(b, d, generator=rng, device=device, dtype=torch.float32)

#         true_ip = Y @ vector          # (b,)
#         approx_ip = Y @ x_f           # (b,)

#         errors[start:end] = approx_ip - true_ip

#     return errors


# n = 128
# a = torch.Tensor([ 2.2500e+00,  2.0781e+00, -3.6250e+00, -1.7344e+00,  9.0234e-01,
#         -8.5156e-01, -2.4062e+00,  9.5312e-01,  8.9844e-01,  6.0156e-01,
#         -4.0312e+00, -5.3711e-02,  2.9688e+00,  7.8125e-03,  3.7812e+00,
#         -2.5781e+00,  3.1562e+00,  2.5195e-01,  4.7070e-01,  8.0078e-01,
#          2.5781e+00, -7.1875e-01,  1.7676e-01,  4.7656e-01, -2.0938e+00,
#         -3.8574e-02,  1.7109e+00, -3.3438e+00, -4.5117e-01, -4.1748e-02,
#         -1.1953e+00,  3.0664e-01,  5.5859e-01, -1.8750e-01,  2.3340e-01,
#         -1.3281e-01,  4.8047e-01, -9.0625e-01, -3.0000e+00, -8.4375e-01,
#          6.1328e-01,  4.3359e-01, -1.7676e-01,  1.4688e+00, -5.0391e-01,
#          6.0547e-01, -8.0469e-01,  9.8438e-01, -6.5625e-01, -1.8750e+01,
#         -5.1562e+00, -4.2812e+00,  7.3750e+00, -7.3750e+00,  2.2344e+00,
#         -1.4438e+01,  1.3250e+01, -2.1750e+01,  5.0000e+00,  8.8125e+00,
#         -3.3250e+01,  1.8625e+01,  4.2250e+01,  3.9531e+00, -4.2812e+00,
#          2.1250e+00, -4.3750e-01,  2.4531e+00, -2.9688e+00, -2.0625e+00,
#          2.2344e+00,  2.2500e+00,  2.5781e+00, -1.1797e+00, -2.7656e+00,
#          4.6680e-01,  9.4141e-01, -3.5312e+00, -7.8125e-01, -2.0312e+00,
#          1.1172e+00, -7.5391e-01,  2.8125e+00,  1.0156e+00,  6.1875e+00,
#          5.2734e-01, -1.6797e+00, -1.0078e+00,  5.9766e-01, -1.3750e+00,
#         -6.4453e-01, -5.4375e+00, -1.7656e+00, -4.1406e-01, -2.6562e-01,
#         -1.5625e-01,  1.5703e+00,  1.1953e+00,  6.0156e-01,  9.5312e-01,
#         -1.1016e+00, -2.3438e-01,  9.5625e+00, -1.1328e+00,  8.0469e-01,
#          1.8750e+00,  1.7031e+00, -2.5938e+00, -4.5117e-01,  9.7266e-01,
#          2.2969e+00,  3.9375e+00,  1.6406e+00, -3.0625e+01,  1.6484e+00,
#         -1.4375e+01,  5.4688e-01,  1.1172e+00,  8.5625e+00,  4.9609e-01,
#          1.0562e+01, -1.7875e+01, -2.5250e+01, -5.3500e+01,  2.3875e+01,
#          5.5750e+01,  2.4750e+01,  8.6000e+01])

# pipeline(a, device=torch.device('cuda'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from TurboQ.TQCache import TurboQuantCache
import gc

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', output_attentions=True).to('cuda')
text = """<think>\nOkay, let's try to solve this problem where I need to find all polynomials P(x, y) such that P(x, y) = P(x, x² - y) for every x and y. Hmm, so the polynomial is invariant when we replace y with x² - y. That seems like a symmetry condition. Let me think about how polynomials can satisfy such a condition.\n\nFirst, maybe I should start by considering the general form of a polynomial in two variables x and y. A polynomial in x and y can be written as a sum of terms like a_{ij}x^i y^j, where a_{ij} are coefficients. The condition given is that replacing y with x² - y doesn't change the polynomial. So if I substitute x² - y in place of y in P(x, y), the resulting polynomial should be the same as the original.\n\nLet me write this out. Let’s suppose P(x, y) = Σ_{i,j} a_{ij} x^i y^j. Then, replacing y by x² - y gives P(x, x² - y) = Σ_{i,j} a_{ij} x^i (x² - y)^j. The equation P(x, y) = P(x, x² - y) implies that each coefficient of x^i y^j in the original polynomial must equal the corresponding coefficient in the transformed polynomial after expansion.\n\nThis might be complicated to handle term by term, but maybe there's a smarter approach. Let me consider that if we substitute y with x² - y, then doing this substitution twice should bring us back to the original y. Because if I replace y with x² - y once, and then do it again, the second substitution would replace y with x² - (x² - y) = y. So applying the substitution twice is the identity transformation. Therefore, this substitution is an involution.\n\nThis suggests that the polynomial can be written in terms of invariants under this substitution. In other words, the polynomial should be expressible using functions (or variables) that are invariant when y is replaced by x² - y. So perhaps if I can find a set of generators for the invariant polynomials under this substitution, then P(x, y) must be a polynomial in those generators.\n\nWhat's an invariant under y ↦ x² - y? Let's see. If I can find some combination of x and y that doesn't change when y is replaced by x² - y. Let me try to find such an invariant. Let's denote the substitution as σ: y → x² - y. Then, applying σ once, y becomes x² - y. Applying σ again, it becomes x² - (x² - y) = y. So σ is indeed an involution.\n\nWhat's fixed by σ? Suppose there's a function f(x, y) such that f(x, y) = f(x, x² - y). Then f is invariant under σ. Let's try to find such a function. For example, consider f(y) = y + (x² - y) = x². Wait, that's just x². Wait, that's a trivial invariant. But actually, if we take the sum y + (x² - y) = x², that's x². But perhaps a better approach is to find something like y*(x² - y) or y + (x² - y) or maybe some combination.\n\nAlternatively, if we think of σ as an involution, the invariants would be polynomials in variables that are fixed by σ. Let’s consider that the substitution σ: y ↦ x² - y. If we can find a variable substitution that makes σ act as a reflection, maybe we can separate variables.\n\nLet’s set z = y - (x² / 2). Then, let's see what σ does to z. If y becomes x² - y, then z becomes (x² - y) - (x² / 2) = x²/2 - y. But z was originally y - x²/2. So under σ, z ↦ -z. Because:\n\nOriginal z: z = y - x²/2\n\nAfter substitution: σ(z) = (x² - y) - x²/2 = x²/2 - y = -(y - x²/2) = -z.\n\nSo z ↦ -z under the substitution. Therefore, any polynomial invariant under σ must be a polynomial in variables that are even in z. Because if you have a term with an odd power of z, it would change sign under σ, which would not preserve the polynomial unless the coefficient is zero.\n\nTherefore, the invariant polynomials under σ are precisely those polynomials that can be expressed in terms of x and z², since z² is invariant (as (-z)^2 = z²). Since z = y - x²/2, then z² = (y - x²/2)^2. Therefore, any polynomial in x and z² would be invariant under σ. Hence, the general solution is that P(x, y) is a polynomial in x and (y - x²/2)^2.\n\nAlternatively, we can write this as P(x, y) = Q(x, (y - x²/2)^2), where Q is any polynomial in two variables. Let me verify this.\n\nSuppose P(x, y) = Q(x, (y - x²/2)^2). Then, substituting y with x² - y gives:\n\nP(x, x² - y) = Q(x, ( (x² - y) - x²/2 )² ) = Q(x, (x²/2 - y)^2 ) = Q(x, (y - x²/2)^2 ) = P(x, y). So yes, this works.\n\nConversely, if P(x, y) is invariant under substitution y ↦ x² - y, then writing z = y - x²/2, so that substitution becomes z ↦ -z, then the invariants are polynomials in z² and x. Hence, P must be a polynomial in x and z², which is exactly the form above. Therefore, the general solution is P(x, y) = Q(x, (y - x²/2)^2), where Q is any polynomial in two variables.\n\nWait, but let me check if there are other invariants. Suppose I try to create another invariant function. Since we have x and z², maybe higher powers or combinations, but since Q is an arbitrary polynomial, that covers all possibilities. So the answer should be all polynomials in x and (y - x²/2)^2.\n\nAlternatively, maybe we can express this in terms of y and x² - y. Let me check. Let me try an example. Suppose Q(x, t) = t, then P(x, y) = (y - x²/2)^2. Let's compute P(x, x² - y) = ( (x² - y) - x²/2 )² = (x²/2 - y)^2 = (y - x²/2)^2 = P(x, y). So yes, it works.\n\nAnother example: if Q(x, t) = x*t, then P(x, y) = x*(y - x²/2)^2. Then P(x, x² - y) = x*( (x² - y) - x²/2 )² = x*(x²/2 - y)^2 = x*(y - x²/2)^2 = same as P(x, y). So that works as well.\n\nWhat if Q is a constant? Then P is constant, which is also invariant. So yes, constants are included.\n\nTherefore, the conclusion is that all such polynomials P(x, y) can be written as polynomials in x and (y - x²/2)^2. So the answer is that P(x, y) is any polynomial in x and (y - x²/2)^2. Therefore, we can write the solution as polynomials in x and y - (x²)/2 squared.\n\nAlternatively, perhaps simplifying (y - x²/2)^2 as y² - x² y + x^4 /4, but since we can write it as that squared term, but in terms of the answer, expressing it in terms of (y - x²/2)^2 is acceptable.\n\nSo, putting it all together, the polynomials satisfying P(x, y) = P(x, x² - y) are exactly those of the form Q(x, (y - x²/2)^2) where Q is a polynomial in two variables. Therefore, the answer is all polynomials in x and (y - (x²)/2)^2.\n\n**Final Answer**\nThe polynomials are those in \\( x \\) and \\( \\left(y - \\frac{x^2}{2}\\right)^2 \\). Thus, the solution is \\boxed{P(x, y) = Q\\left(x, \\left(y - \\frac{x^2}{2}\\right)^2\\right)} where \\( Q \\) is any polynomial in two variables.\n</think>\n\nTo solve the problem of finding all polynomials \\( P(x, y) \\) such that \\( P(x, y) = P(x, x^2 - y) \\) for every \\( x \\) and \\( y \\), we start by considering the symmetry condition imposed by the substitution \\( y \\mapsto x^2 - y \\). This substitution is an involution, meaning applying it twice returns the original value. \n\nWe introduce the variable \\( z = y - \\frac{x^2}{2} \\). Under the substitution \\( y \\mapsto x^2 - y \\), \\( z \\) transforms as follows:\n\\[\nz \\mapsto (x^2 - y) - \\frac{x^2}{2} = \\frac{x^2}{2} - y = - \\left( y - \\frac{x^2}{2} \\right) = -z.\n\\]\nThus, \\( z \\mapsto -z \\) under the substitution. For a polynomial to be invariant under this substitution, it must be even in \\( z \\). Therefore, the polynomial can be expressed in terms of \\( x \\) and \\( z^2 \\).\n\nSince \\( z = y - \\frac{x^2}{2} \\), we have \\( z^2 = \\left( y - \\frac{x^2}{2} \\right)^2 \\). Hence, any polynomial invariant under the substitution \\( y \\mapsto x^2 - y \\) must be a polynomial in \\( x \\) and \\( \\left( y - \\frac{x^2}{2} \\right)^2 \\).\n\nThus, the solution is all polynomials \\( P(x, y) \\) that can be written in the form:\n\\[\nP(x, y) = Q\\left( x, \\left( y - \\frac{x^2}{2} \\right)^2 \\right),\n\\]\nwhere \\( Q \\) is any polynomial in two variables.\n\n\\[\n\\boxed{P(x, y) = Q\\left(x, \\left(y - \\frac{x^2}{2}\\right)^2\\right)}"""
tokenized_text = tokenizer(text, return_tensors='pt')


crazy = None

def inference(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    tokenized_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_promt=True
    )

    TurboCache = TurboQuantCache(
        device=model.device, 
        num_key_heads=model.config.num_key_value_heads,
        dim=128,
        config=model.config
    )

    encoded = tokenizer(text, return_tensors='pt').to('cuda')
    model.eval()
    print("Starting inference\n")
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=500,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            past_key_values=TurboCache,
            use_cache=True, 
            return_dict_in_generate=True,
        )
        print(type(output.past_key_values))
        print(output.past_key_values)
        print(f"Final tokens: {output}")

        global crazy
        crazy = output.past_key_values    
        return tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

output = inference(text)