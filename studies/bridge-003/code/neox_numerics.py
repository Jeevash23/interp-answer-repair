"""Align unprocessed NeoX float32 arithmetic with the original HF implementation.

No weights or mathematical operations are changed. HF uses native layer_norm and
(MLP + attention) + residual; TransformerLens defaults to a manually reduced
layer norm and (residual + attention) + MLP. Large NeoX residual coordinates can
make their float32 rounding differences visible in final logits.
"""

from types import MethodType

import torch
import torch.nn.functional as F


def layer_norm(self, x):
    return F.layer_norm(x, (x.shape[-1],), self.w, self.b, self.eps)


def fused_qkv(self, query_input, key_input, value_input):
    assert query_input is key_input and query_input is value_input
    qkv = F.linear(query_input, self._neox_qkv_weight, self._neox_qkv_bias)
    qkv = qkv.view(*query_input.shape[:2], self.cfg.n_heads, 3 * self.cfg.d_head)
    q, k, v = qkv.chunk(3, dim=-1)
    return self.hook_q(q), self.hook_k(k), self.hook_v(v)


def mlp_forward(self, x):
    pre = self.hook_pre(F.linear(x, self._neox_in_weight, self.b_in))
    post = self.hook_post(self.act_fn(pre))
    return F.linear(post, self._neox_out_weight, self.b_out)


def unembed_forward(self, x):
    assert torch.count_nonzero(self.b_U) == 0
    return F.linear(x, self._neox_unembed_weight)


def attention_forward(
    self, query_input, key_input, value_input, past_kv_cache_entry=None, attention_mask=None
):
    from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

    assert past_kv_cache_entry is None and attention_mask is None
    q, k, v = self.calculate_qkv_matrices(query_input, key_input, value_input)
    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    n = q.shape[-2]
    q, k = apply_rotary_pos_emb(q, k, self.rotary_cos[None, :n], self.rotary_sin[None, :n])
    q = self.hook_rot_q(q.transpose(1, 2)).transpose(1, 2)
    k = self.hook_rot_k(k.transpose(1, 2)).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(2, 3)) * (self.cfg.d_head**-0.5)
    mask = torch.full((1, 1, n, n), torch.finfo(scores.dtype).min, device=scores.device).triu(1)
    scores = self.hook_attn_scores(scores + mask)
    pattern = self.hook_pattern(F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype))
    z = self.hook_z(torch.matmul(pattern, v).transpose(1, 2).contiguous())
    weight = self.W_O.permute(2, 0, 1).reshape(self.cfg.d_model, -1).contiguous()
    out = F.linear(z.reshape(*z.shape[:2], -1), weight, self.b_O)
    if self.cfg.use_attn_result:
        result = torch.einsum("bphd,hdm->bphm", z, self.W_O)
        out = out + (self.hook_result(result) - result).sum(2)
    return out


def block_forward(
    self, resid_pre, shortformer_pos_embed=None, past_kv_cache_entry=None, attention_mask=None
):
    assert shortformer_pos_embed is None
    assert self.cfg.parallel_attn_mlp and not self.cfg.use_split_qkv_input
    assert not self.cfg.use_attn_in and not self.cfg.use_hook_mlp_in
    resid = self.hook_resid_pre(resid_pre)
    norm = self.ln1(resid)
    attn = self.hook_attn_out(
        self.attn(
            query_input=norm,
            key_input=norm,
            value_input=norm,
            past_kv_cache_entry=past_kv_cache_entry,
            attention_mask=attention_mask,
        )
    )
    mlp = self.apply_mlp(self.ln2(resid))
    return self.hook_resid_post(mlp + attn + resid)


def align(model, raw):
    for i, block in enumerate(model.blocks):
        block.forward = MethodType(block_forward, block)
        block.ln1.forward = MethodType(layer_norm, block.ln1)
        block.ln2.forward = MethodType(layer_norm, block.ln2)
        block.mlp.register_buffer(
            "_neox_in_weight", raw.gpt_neox.layers[i].mlp.dense_h_to_4h.weight.detach()
        )
        block.mlp.register_buffer(
            "_neox_out_weight", raw.gpt_neox.layers[i].mlp.dense_4h_to_h.weight.detach()
        )
        block.mlp.forward = MethodType(mlp_forward, block.mlp)
        projection = raw.gpt_neox.layers[i].attention.query_key_value
        block.attn.register_buffer("_neox_qkv_weight", projection.weight.detach().clone())
        block.attn.register_buffer("_neox_qkv_bias", projection.bias.detach().clone())
        block.attn.calculate_qkv_matrices = MethodType(fused_qkv, block.attn)
        block.attn.forward = MethodType(attention_forward, block.attn)
    model.ln_final.forward = MethodType(layer_norm, model.ln_final)
    model.unembed.register_buffer("_neox_unembed_weight", raw.lm_head.weight.detach())
    model.unembed.forward = MethodType(unembed_forward, model.unembed)


def align_rotary(model, raw, device):
    positions = torch.arange(model.cfg.n_ctx, device=device)[None]
    dummy = torch.zeros(1, 1, model.cfg.d_model, device=device)
    cos, sin = raw.gpt_neox.rotary_emb(dummy, positions)
    for block in model.blocks:
        block.attn.rotary_cos = cos[0].clone()
        block.attn.rotary_sin = sin[0].clone()
