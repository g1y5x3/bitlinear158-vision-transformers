# BitLinear-Vision-Transformers
Since the paper [The Era of 1-bit LLMs](https://arxiv.org/pdf/2402.17764v1.pdf) was relased, it makes me wonder whether training transformers with
the proposed `BitLinear` can also work across all modality on applications other than LLMs, for example, vision based models such as
~~ViT~~([TerViT](https://arxiv.org/abs/2201.08050) but no source code that I can find), DETR, DINO, LlaVa etc.

## DETR (Detection Transformer)
After some attempts to modify DETR base on some of the most popular computer vision libraries such as __ultralytics__, __mmdet__, __detectron2__, it
felt like I was editing _yaml_ files most of the time which was quite frustrating. The implementation from huggingface seems more straight forward
but the code also looks very similar to [original detr repo](https://github.com/facebookresearch/detr), which was a bit out dated, e.g, it didn't
support mixed precision, it has low GPU utilization during training. Moreoever, it felt a bit too complex for an idea that is fairly straight forward.
Therefore, I decided to rewrite everything with the goal to make it easy to read, study, build and hack around.
## Notes on BitLinear
$f(x)=\tilde{W}\tilde{x}$

$\tilde{W} = {RoundClip}(\dfrac{W}{\beta + \epsilon}, -1, 1)$, where $RoundClip(x, a, b)=max(a, min(b, round(x)))$, and 
$\displaystyle \beta = \frac{1}{nm}\sum_{ij}|W_{ij}|$.

$\tilde{x} = {Clip}(\dfrac{xQ_b}{\gamma}, -Q_b+\epsilon, Q_b-\epsilon)$, where $Clip(x, a, b)=max(a, min(b, x))$, and $\gamma = ||x||_{\infty}$

First of all, both `x` and `w` are still in `float16` during training. However, they do get __quantized__ to maintain the property of 8 bits for `x` 
and ternary for `w`. In the code provided by 
[FAQ](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf),
both `x_quant` and `w_quant` are also __rescaled__ before `F.linear` which becomes

$f(x)=(\beta\tilde{W})(\dfrac{\gamma\tilde{x}}{Q_b})$

```python
x_scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
quant   = (x * x_scale).round().clamp(-128, 127) / x_scale
x_quant = x + (quant - x).detach()

w_scale = 1.0 / w.abs().mean().clamp(min=1e-5)
quant   = (w * w_scale).round().clamp(-1, 1) / w_scale
w_quant = w + (quant - w).detach()

output  = F.linear(x_quant, w_quant)
```
Using `x + (quant - x).detach()` and `w + (quant - w).detach()` is a trick to employ straight-through estimator that makes `F.linear` think it is
still calculating $f(x)=Wx$ instead of $\tilde{W}\tilde{x}$, which can bypass the non-differentiable functions ($RoundClip$ and $Clip$), therefore
resulting the gradient $\nabla f = \dfrac{\partial f}{\partial W} = x$.

But mathmatically this calculation is also equivalent to
$f(x)=(\beta\tilde{W})(\dfrac{\gamma\tilde{x}}{Q_b})=\tilde{W}\tilde{x}(\dfrac{\beta\gamma}{Q_b})$

The implementation can then become
```python
x_scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
quant   = (x * x_scale).round().clamp(-128, 127)
x_quant = x + (quant - x).detach()

w_scale = 1.0 / w.abs().mean().clamp(min=1e-5)
quant   = (w * w_scale).round().clamp(-1, 1)
w_quant = w + (quant - w).detach()

output  = F.linear(x_quant, w_quant) / (x_scale * w_scale)
```
where `x_quant` and `w_quant` are $[-127, 128]$ (8 bits) and $\{-1, 0, 1\}$ (tenary) which means in theory this matmul can be done very easily in 
much lower precision ([INT8 GEMM](https://github.com/jundaf2/CUDA-INT8-GEMM)?). 

However, due to floating-point arithmetic operations are not always associative or commutative, the resulting losses are slightly different (a few 
more tests in [models/bitlinear.py](models/bitlinear.py#L59-L83)).

![when-to-rescale](figures/simple_experiments.png)

## Reults

## TODO
- [x] rewrite the model to make the coder simplier, more readable, and easy to study.
    - [x] implement `MultiheadAttention` from scratch but keep `F.scaled_dot_product_attention` to utilized the optimized flash attentions kernel.
    - [x] remove the entirety of `NestedTensor` in DETR, the forward pass now takes two arguments both padded img and padding mask 
    - [x] simply SetCriterion, only `l1_loss`, `giou_loss`, and `cross_entropy` were used to compute the gradients (this is the slowest part). 
    - [x] change from post-LayerNorm to pre-LayerNorm for easy intergration with BitLinear
    - [x] training in float16 using `amp`
    - [ ] deepspeed integration for multigpu training
- [ ] perform a full COCO training run with `nn.Linear` vs `BitLinear`
- [ ] Build custom cuda kernel for `BitLinear`. According to the 
      [FAQ](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf),
      with FP8 GEMM kernels, we can use FP8 activations for training and quantize to INT8 for inference.
- [ ] Try `BitLinear` on DINO, LlaVa.