## Notes
1. This implementation was based on [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr) with the goal to keep things 
as minimal as possible. Personally, I don't like to have multiple `build_x` functions to initiate a class constructor of the model, I would much 
rather prefer to initialize the constructors directly.

2. Since all `q`, `k`, and `v` tensors are being reshaped to `[bsz, num_heads, seq_len, head_dim]` right before `scaled_dot_product_attention` in 
torch, this implementation sticks with `bsz` as all tensors first dim for better clarity.

3. There is only one `mask` that really matters for this model which is the `src_padding_mask` created during batch collection. The rest of input 
arguments during forward pass were thus thrown away to keep things simple.

4. Originally the transformer was used post-LayerNorm in both of its encoders and decoders. Since then, it was first suggested by 
[this paper](https://arxiv.org/pdf/2002.04745v1.pdf) that pre-LayerNorm Transformer can reach comparable results with less training time. 
Additionally, this change makes the adoptation of BitLinear (norm -> linear -> activation) module much easier.