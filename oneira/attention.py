"""Chunk-wise causal text attention: video rows of chunk k do not see captions of chunks after k."""

import torch

import diffsynth.models.minimax_h3_dit as h3_dit
from diffsynth.core.attention import attention_forward

_state = {"total": -1, "groups": []}
_orig_attention = None


def install():
    """Patch the DiT attention. Call before the pipeline is created."""
    global _orig_attention
    if _orig_attention is not None:
        return
    _orig_attention = h3_dit._sdpa_varlen_attention

    def masked_attention(q, k, v, cu_seqlens, softmax_scale):
        out = _orig_attention(q, k, v, cu_seqlens, softmax_scale)
        if q.shape[0] != _state["total"]:
            return out
        # recompute the masked query rows on the allowed keys only
        for q_idx, keep in _state["groups"]:
            sq = q[q_idx].transpose(0, 1).unsqueeze(0)
            sk = k[keep].transpose(0, 1).unsqueeze(0)
            sv = v[keep].transpose(0, 1).unsqueeze(0)
            so = attention_forward(sq, sk, sv, scale=softmax_scale)
            out[q_idx] = so.squeeze(0).transpose(0, 1).to(out.dtype)
        return out

    h3_dit._sdpa_varlen_attention = masked_attention


def set_groups(total, groups):
    _state["total"], _state["groups"] = total, groups


def clear():
    _state["total"], _state["groups"] = -1, []


def build_groups(packed, spans, ref_rows, n_lat, n_chunk, chunk_lat, frame_rows):
    """Returns (total length, [(query rows, allowed keys)]) for every chunk except the last."""
    video_pos = packed["img_pos"][ref_rows:]
    assert video_pos.shape[0] == n_lat * frame_rows, video_pos.shape
    used = int(packed["cu_seqlens"][1])
    groups = []
    for c in range(n_chunk - 1):
        keep = torch.ones(used, dtype=torch.bool)
        for j in range(c + 1, n_chunk):
            keep[spans[j][0] : spans[j][1]] = False
        rows = video_pos[c * chunk_lat * frame_rows : (c + 1) * chunk_lat * frame_rows]
        groups.append((rows, torch.nonzero(keep, as_tuple=False).squeeze(1)))
    return int(packed["cu_seqlens"][-1]), groups
