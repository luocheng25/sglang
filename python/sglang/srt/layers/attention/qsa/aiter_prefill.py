"""gfx942 QSA prefill attention through aiter's ``qsa_sparse_prefill``.

A drop-in for ``PyHIPAttention`` with the same three methods, selected by
SGLANG_USE_AITER_QSA_PREFILL; the PyHIP indexer is unaffected.
"""

from __future__ import annotations

import inspect
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.pyhip_backend import PyHIPAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_exec

logger = logging.getLogger(__name__)


class AiterQSAAttention:
    def __init__(self, *, kv_cache_dtype):
        from aiter.ops.qsa import qsa_sparse_prefill

        self.attention = qsa_sparse_prefill
        parameters = inspect.signature(qsa_sparse_prefill).parameters
        self.validate = envs.SGLANG_TEST_PYHIP_QSA.get()
        # FP8 K/V: FP8 MFMA (FP8 Q and probabilities), or dequantized to BF16.
        self.fp8_compute = "fp8" if envs.SGLANG_OPT_PYHIP_QSA_FP8_MFMA.get() else "bf16"
        # FP8 K/V arrive with aiter's FP8 port, which adds k_scale/v_scale/compute.
        self.fp8_kv_unsupported = (
            kv_cache_dtype == torch.float8_e4m3fnuz and "k_scale" not in parameters
        )
        # batch_invariant=True makes every numerical routing decision per request.
        self.batch_invariant = {}
        if get_exec().deterministic.enable_deterministic_inference:
            if "batch_invariant" in parameters:
                self.batch_invariant = dict(batch_invariant=True)
                logger.info(
                    "--enable-deterministic-inference: aiter QSA prefill attention "
                    "passes batch_invariant=True"
                )
            else:
                logger.warning(
                    "--enable-deterministic-inference: the installed aiter "
                    "qsa_sparse_prefill takes no batch_invariant, so QSA prefill "
                    "attention still routes by the whole batch"
                )
        self.checked_layouts = set()
        self.served_layers = set()
        self.declined_layers = set()

    def can_use_for_forward(self, *, q, k, v, indices, layer, forward_mode, kwargs):
        # As with PyHIP, only EXTEND qualifies; other modes stay native without a log.
        if forward_mode != ForwardMode.EXTEND:
            return False
        if self.fp8_kv_unsupported:
            reason = (
                "the installed aiter qsa_sparse_prefill takes no FP8 K/V "
                "(no k_scale/v_scale)"
            )
        elif not PyHIPAttention.can_use_for_forward(
            q=q,
            k=k,
            v=v,
            indices=indices,
            layer=layer,
            forward_mode=forward_mode,
            kwargs=kwargs,
        ):
            reason = (
                f"unsupported inputs or graph/compile state: q {tuple(q.shape)} "
                f"{q.dtype}, k {tuple(k.shape)} {k.dtype}, indices "
                f"{tuple(indices.shape)}, heads {layer.tp_q_head_num}/"
                f"{layer.tp_k_head_num}"
            )
        else:
            return True
        if layer.layer_id not in self.declined_layers:
            self.declined_layers.add(layer.layer_id)
            logger.warning(
                "aiter QSA prefill attention declines layer %s, which falls back "
                "to SGLang's kernel: %s",
                layer.layer_id,
                reason,
            )
        return False

    # aiter's qsa_sparse_prefill reads a K/V pool in place under PyHIP's contract.
    can_read_kv_slots = staticmethod(PyHIPAttention.can_read_kv_slots)

    def forward(
        self,
        *,
        q,
        k,
        v,
        indices,
        query_lens,
        prefix_lens,
        scale,
        layer_id,
        kv_slots=None,
        k_scale=None,
        v_scale=None,
    ):
        # With kv_slots, k/v are the layer's token pool and kv_slots maps packed context tokens to it.
        fp8 = {}
        if k_scale is not None:
            fp8 = dict(k_scale=k_scale, v_scale=v_scale, compute=self.fp8_compute)
        output = self.attention(
            q,
            k,
            v,
            indices,
            query_lens=query_lens,
            prefix_lens=prefix_lens,
            softmax_scale=scale,
            kv_slots=kv_slots,
            **self.batch_invariant,
            **fp8,
        )
        if layer_id not in self.served_layers:
            self.served_layers.add(layer_id)
            logger.info(
                "aiter QSA prefill attention serves layer %s "
                "(K/V %s%s, batch_invariant=%s)",
                layer_id,
                k.dtype,
                f", compute={fp8['compute']}" if fp8 else "",
                bool(self.batch_invariant),
            )
        layout = (layer_id, tuple(query_lens), tuple(prefix_lens), q.shape[1], scale)
        if self.validate and layout not in self.checked_layouts:
            from sglang.test.qsa_validation import check_attention

            if kv_slots is not None:
                k, v = k.index_select(0, kv_slots), v.index_select(0, kv_slots)
            if fp8:
                # The reference reads K/V in Q's dtype; FP8 -> BF16 is exact and the
                # FP32 scale product rounds once (exact for unit scales).
                k = k.to(q.dtype).mul_(k_scale)
                v = v.to(q.dtype).mul_(v_scale)
            stats = check_attention(
                q=q,
                k=k,
                v=v,
                indices=indices,
                output=output,
                query_lens=query_lens,
                prefix_lens=prefix_lens,
                scale=scale,
                fp8_compute=fp8.get("compute") == "fp8",
            )
            self.checked_layouts.add(layout)
            logger.info(
                "aiter QSA attention checked: layer=%s queries=%s prefixes=%s%s",
                layer_id,
                query_lens,
                prefix_lens,
                "" if stats is None else f" fp8={stats}",
            )
        return output
