import os
import sys
from pathlib import Path
from typing import Optional

import torch


_FLASH_ATTN_VECBIAS_MODULE = None
_FLASH_ATTN_VECBIAS_IMPORT_ERROR: Optional[Exception] = None


def _load_vecbias_module():
    global _FLASH_ATTN_VECBIAS_MODULE, _FLASH_ATTN_VECBIAS_IMPORT_ERROR
    if _FLASH_ATTN_VECBIAS_MODULE is not None or _FLASH_ATTN_VECBIAS_IMPORT_ERROR is not None:
        return _FLASH_ATTN_VECBIAS_MODULE

    repo_root = Path(__file__).resolve().parents[4]
    ext_root = repo_root / "flash-attention-src"
    if str(ext_root) not in sys.path:
        sys.path.insert(0, str(ext_root))

    try:
        import flash_attn_vecbias_cuda as mod

        _FLASH_ATTN_VECBIAS_MODULE = mod
    except Exception as exc:  # pragma: no cover - best effort optional import
        _FLASH_ATTN_VECBIAS_IMPORT_ERROR = exc

    return _FLASH_ATTN_VECBIAS_MODULE


def flash_attn_vecbias_available() -> bool:
    if os.getenv("LLAMAFACTORY_DISABLE_VECBIAS_KERNEL", "0") == "1":
        return False
    return _load_vecbias_module() is not None


def can_use_flash_attn_vecbias(
    query_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    sliding_window: Optional[int],
) -> bool:
    return (
        flash_attn_vecbias_available()
        and query_states.is_cuda
        and query_states.dtype == torch.bfloat16
        and query_states.size(-1) == 64
        and attention_mask is None
        and dropout_p == 0.0
        and is_causal
        and sliding_window is None
    )


class _FlashAttnVecBiasFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        vector_bias: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        mod = _load_vecbias_module()
        out, softmax_lse, _, rng_state = mod.fwd_vecbias(
            q,
            k,
            v,
            None,
            None,
            vector_bias,
            0.0,
            softmax_scale,
            True,
            -1,
            -1,
            0.0,
            False,
            None,
        )
        ctx.softmax_scale = softmax_scale
        ctx.save_for_backward(q, k, v, out, softmax_lse, vector_bias, rng_state)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        mod = _load_vecbias_module()
        q, k, v, out, softmax_lse, vector_bias, rng_state = ctx.saved_tensors
        dq, dk, dv, _, dvector_bias = mod.bwd_vecbias(
            dout.contiguous(),
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,
            None,
            None,
            vector_bias,
            0.0,
            ctx.softmax_scale,
            True,
            -1,
            -1,
            0.0,
            False,
            None,
            rng_state,
        )
        return dq, dk, dv, dvector_bias, None


def flash_attn_vecbias(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    vector_bias: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    vector_bias = vector_bias.to(dtype=torch.float32).contiguous()
    return _FlashAttnVecBiasFn.apply(q, k, v, vector_bias, softmax_scale)
