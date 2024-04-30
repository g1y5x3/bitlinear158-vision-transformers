"""
Implementation Reference
https://github.com/joey00072/ohara/blob/master/experiments/bitnet/bitnet.py
https://huggingface.co/NousResearch/OLMo-Bitnet-1B/blob/main/model.py

Paper Reference
BitNet:                https://arxiv.org/pdf/2310.11453.pdf
The Era of 1-bit LLMs: https://arxiv.org/pdf/2402.17764v1.pdf
FAQ:                   https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
"""

import torch, bitblas
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

# Bitblas does not support autograd yet
class BitBLASLinear(nn.Linear):
  def __init__(self, in_features:int, out_features: int, bias: bool=True):
    super(BitLinear, self).__init__(in_features, out_features, bias)
    matmul_config = bitblas.MatmulConfig(
      N=self.out_features,  # N dimension
      K=self.in_features,   # K dimension
      A_dtype="float16",  # activation A dtype
      W_dtype="int2",  # weight W dtype
      accum_dtype="float16",  # accumulation dtype
      out_dtype="float16",    # output dtype
      layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
      with_bias=bias,  # bias
      # configs for weight only quantization
      group_size=None,  # setting for grouped quantization
      with_scaling=False,  # setting for scaling factor
      with_zeros=False,  # setting for zeros
      zeros_mode=None,  # setting for how to calculating zeros
    )
    self.matmul = bitblas.Matmul(config=matmul_config)
  
  def weight_quant(self, w: Tensor):
    w = w.float()
    s = 1 / w.abs().mean().clamp(min=1e-5)
    w_quant = (w*s).round().clamp(-1,1)
    return w_quant.type(torch.int8), s

  def forward(self, x: Tensor) -> Tensor:
    w = self.weight
    w_quant, w_scale = self.weight_quant(w)
    output = self.matmul(x, w_quant) / w_scale
    return output

class BitLinear(nn.Linear):
  # FP16 x INT1.58 (simulated by FP16)
  def __init__(self, in_features:int, out_features: int, bias: bool=True):
    super(BitLinear, self).__init__(in_features, out_features, bias)

  def activation_quant(self, x: Tensor):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    quant = (x * scale).round().clamp(-128, 127)
    return quant, scale

  def weight_quant(self, w: Tensor):
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    quant = (w * scale).round().clamp(-1, 1)
    return quant, scale

  def forward(self, x: Tensor) -> Tensor:
    w = self.weight
    w_quant, w_scale = self.weight_quant(w)
    output = F.linear(x, w + (w_quant - w).detach()) / w_scale
    return output

if __name__ == "__main__":
  # a minimal example to demonstrate the difference between pre-rescale and post-rescale without having to worry about gradients
  bsz, seq_len, d = 2, 8, 16

  def pre_scale_vs_post_scale(dtype, device):
    print(f"precision: {dtype}, device: {device}")

    x = torch.randn((bsz, seq_len, d), dtype=dtype).to(device)
    w = torch.randn((d, d), dtype=dtype).to(device)

    x_scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    x_quant = (x * x_scale).round().clamp(-128, 127)

    w_scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    w_quant = (w * w_scale).round().clamp(-1, 1)

    output1 = F.linear(x_quant/x_scale, w_quant/w_scale)
    output2 = F.linear(x_quant, w_quant)/(x_scale*w_scale)
 
    print(f"rtol=1e-05, atol=1e-05 {torch.allclose(output1, output2, rtol=1e-05, atol=1e-05, equal_nan=True)}")
    print(f"rtol=1e-03, atol=1e-03 {torch.allclose(output1, output2, rtol=1e-03, atol=1e-03, equal_nan=True)}")
    print(f"rtol=1e-03, atol=1e-02 {torch.allclose(output1, output2, rtol=1e-03, atol=1e-02, equal_nan=True)}")

  pre_scale_vs_post_scale(torch.float16, "cpu")
  pre_scale_vs_post_scale(torch.float16, "cuda")