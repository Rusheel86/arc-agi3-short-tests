#!/usr/bin/env python3
"""E083: K8V4 main KV cache (fp8 keys, 4-bit nvfp4 values) for the QSA layers of our vLLM build.

PLAIN WORDS
    Our server keeps every game's attention memory (the KV cache) in 8-bit (fp8). E062
    put both halves of it (keys and values) in 4 bits; that held 1.62x more tokens but
    cost too much quality (prompt NLL 0.047 vs 0.039, E061b). This script keeps the
    keys in 8 bits exactly as today and stores only the values in 4 bits. A token then
    takes 800 instead of 1,024 bytes per attention layer: the block grows from 1,664 to
    2,128 tokens at almost the same bytes, and the same GPU memory holds ~1.27x more
    tokens. It edits three Python files of the installed vLLM package before the server
    starts and changes nothing unless the server runs with --kv-cache-dtype fp8 (or
    fp8_e4m3) AND the environment variable ARC3_KVQ_K8V4=1.

PROVENANCE AND LICENCE
    The nvfp4 bit helpers (E2M1 decode/encode, E4M3 scale decode/encode) and the value
    quantiser are andreasgru's (vllm-qwen38-flash-next-kv-quant, kvq-patch.py,
    Apache-2.0, Copyright andreasgru; vendor/kv-quant), after vllm-project/vllm #44389
    and #54846, taken from our E062 port (experiments/e062_kvq_port.py) and renamed.
    The mixed fp8-K / nvfp4-V row layout, its writer and the kernel's V-only branch are
    new here. Apache-2.0, like vLLM. Everything inserted into vLLM carries the marker
    ``ARC3-KVQ-E083``.

LAYOUT (one row per token and KV head, D = head_dim = 256)
    [ K e4m3: D bytes | V e2m1: D // 2 bytes (two nibbles per byte, even index in the
      low nibble) | V e4m3 block scales: D // 16 bytes, one per 16 values, linear ]
    = 25 * D / 16 = 400 bytes per (token, head); 2 KV heads -> 800 B/token/layer.
    The row is self-contained (no scale swizzle across tokens, no K/V head slots), so
    it is correct under every KV cache layout (BLNHC, BLHNC, ...): no layout forcing,
    unlike E062. Keys are quantised exactly as reshape_and_cache_flash does for fp8
    (x / k_scale, round-to-nearest-even, saturating at 448), so the key bytes are the
    same bytes today's fp8 cache holds.

WHAT IS PORTED
    vllm/models/qwen4_exp/nvidia/qsa.py      flag ARC3_KVQ_K8V4, K8V4 page geometry
        (customize_spec + owner spec, idempotent, fp8-per-tensor specs only), K8V4
        writer dispatch, K8V4 views in forward_qsa, misconfiguration guards.
    vllm/models/qwen4_exp/nvidia/ops/qsa.py  V-only nvfp4 read branch (V_NVFP4
        constexpr; keys stay on the fp8 path), K8V4 SM120 tile table + stage rule,
        wrapper checks, warmup that recognises a K8V4 cache (the stock warmup would
        treat its 400-byte rows as fp8 with HEAD_DIM 200 and fail at startup), Triton
        writer, cache views.
    vllm/platforms/interface.py              hybrid block alignment uses the K8V4 row
        (800 B/token/layer -> block 2,128 tokens) whichever non-SSM backend is found
        first (the QSA side cache does not know the packing).

SAFETY
    * Pinned: every target file and every file whose behaviour the port relies on must
      have its pinned sha256 (vLLM e975732 + overlay e3a6fe0d, LF endings, as in the
      image). Anything else -> refuse, nothing written. The E062 port and this one edit
      the same stock files and exclude each other (each refuses the other's output).
    * Atomic: all pins and all anchors (exactly one match each) are checked and all new
      file contents are built and compiled in memory before the first write; a failed
      write rolls back the files already replaced.
    * Idempotent: a second run finds the pinned patched contents and reports
      "already applied" without touching anything.
    * Stale bytecode of the patched modules is removed so the new source is used.
    * Off = stock: with ARC3_KVQ_K8V4 unset or 0 the spec, block size, writer and
      kernel outputs are the stock ones (bf16/fp8 kernel outputs bitwise equal in the
      CPU tests). On: the per-step path adds strided views (metadata only) and one
      Triton writer launch whose grid depends only on the slot-mapping length; no host
      sync, no data-dependent Python branch, so CUDA-graph capture sees fixed launches.

USAGE
    python e083_k8v4_port.py --site <dir containing vllm/>   (or env ARC3_KVQ_SITE)
        [--overlay-identity <runtime dataset>/PATCH_IDENTITY.json] [--check]
    exit 0 = applied or already applied; 2 = pins differ; 3 = anchors differ;
    4 = write failed and was rolled back. Nothing is changed unless the exit code is 0.

    python e083_k8v4_port.py --selftest [--out verdict.json]
        GPU kernel self-test inside the server's environment (see selftest()). Exit 0
        only when every check passes; on any other exit start the server in plain fp8.

    python e083_k8v4_port.py --server-check http://127.0.0.1:PORT --model NAME
        [--baseline fp8_check.json] [--nll-max 0.043] [--out verdict.json]
        Post-start check of the running K8V4 server (needle recall, prompt NLL, MTP
        acceptance). Restart in plain fp8 on a non-zero exit.

SERVER SETTINGS FOR K8V4
    ARC3_KVQ_K8V4=1 in the server's environment, --kv-cache-dtype fp8_e4m3, and a
    prefix-match unit that divides the 2,128-token block: 112 or 152 (128 does not;
    vLLM refuses to start with it). With 4 draft tokens the block is 2,160 and the unit
    must be 120 or 144. Expect "Setting attention block size to 2128" in the log.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

MARKER = "ARC3-KVQ-E083"
OTHER_PORT_MARKER = "ARC3-KVQ-E062"
FLAG_ENV = "ARC3_KVQ_K8V4"
VLLM_COMMIT = "e9757321527ca1ecd514c07c1418dd2c53da3d19"
VLLM_VERSION_TAG = "ge97573215"
OVERLAY_SHA256 = "e3a6fe0d9f010bc1fb66e53e7dc6fde43272824d5200da5a49b5c4456065e0cb"

QSA = "vllm/models/qwen4_exp/nvidia/qsa.py"
OPS = "vllm/models/qwen4_exp/nvidia/ops/qsa.py"
PLAT = "vllm/platforms/interface.py"

# sha256 of the files as installed (vLLM e975732 image + ARC3 overlay e3a6fe0d, LF).
# The three targets are stock vLLM files (not overlay members); same bytes E062 pins.
TARGET_PINS = {
    QSA: "f7307663d341e3fdf5ee2ae866aaea449f1e89564938bffd66fb3211147b4f3f",
    OPS: "6a41ab3e8af9f6579dd64c7c5b08364f2df558061a1d2a57cef74cb5c2ea18bb",
    PLAT: "70613f8d77b76d79290b81f8c1f7f53735e88f479802a9157af765217a47b2d6",
}
# Read-only files whose behaviour the port depends on (spec -> page size, view
# construction, block sizes and prefix-match unit, side-cache ring, MTP wiring, the
# parent writer, the warmup driver). Overlay members carry the overlay's bytes.
DEPENDENCY_PINS = {
    "vllm/models/qwen4_exp/common/qsa_cache.py": "f2e478b9ceb73fe0d10d090996290aaa8a8be210858607dc7d49453f71d60ec5",
    "vllm/models/qwen4_exp/nvidia/mtp.py": "6a7db22e742c4d4287b17ebc35e38d61b97f04441639d4aea3fddd2392ebe0c3",
    "vllm/models/qwen4_exp/nvidia/model.py": "2e03b2faa3eacf7bd6cb5b1a081240b7efc586d5eac87144104fc8aea1168dad",
    "vllm/v1/worker/gpu_model_runner.py": "a2d3709963d463256b8b62ecb5264bce10e1b133d552ab10afdd31f68b42d1b2",
    "vllm/v1/core/kv_cache_utils.py": "816aea33eaa28a15b8d739be602a877aa0be1a8ec83853cdc92412dcdd137b4e",
    "vllm/v1/kv_cache_interface.py": "9ec325175c6d65a466319a67b61afe4f62fc5639ae1aaa1575c033aa735c7943",
    "vllm/v1/kv_cache_layout.py": "7a32e95b61238e4e2fd8b0c420830565002af3275ca2eded6d78bc60dcaf7df0",
    "vllm/v1/attention/backends/utils.py": "ff92a61b8a0439539cc8cf9dd353efc29765d8094a4948b4824188de0bde9063",
    "vllm/utils/torch_utils.py": "31e8a695f7efa52584ce20cc33edfb643b70c28ce83d273feb34326c61ab631c",
    "vllm/v1/worker/utils.py": "3351a0410ac1c1635ece84881679e491df5cd531b0b3d419f0a1990af5896892",
    "vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py": "ce5fdfb4a0dc08d02d266740f629fc383ae1624a89a5c7414a669121c98f3f57",
    # new vs E062: the parent impl (ctor probe, inherited fp8 writer) and the
    # warmup tensor helper the patched warmup builds its K8V4 requests with
    "vllm/v1/attention/backends/flash_attn.py": "0598232cb726e8ce97ec1db725a6558218e51b7a8051abd7d74193d2283cacb1",
    "vllm/model_executor/warmup/jit_warmup_triton_helper.py": "5f521c52cd12453653826b8377ecdf508862209384cb62e30832d3c313cf4aca",
}
# Facts in the pinned dependencies that the port relies on, re-checked as text so a
# reader of this script can see them (each must occur exactly once).
DEPENDENCY_FACTS = [
    # a packed spec's cell size is what the allocator uses
    (
        "vllm/v1/kv_cache_interface.py",
        "        if self.state_content_bytes is not None:\n"
        "            return self.state_content_bytes\n",
    ),
    ("vllm/v1/kv_cache_interface.py", "    FP8_PER_TENSOR = 1  # per-tensor scales (current fp8 path)\n"),
    # fp8 caches are uint8 storage
    ("vllm/utils/torch_utils.py", '    "fp8_e4m3": torch.uint8,\n'),
    # every attention spec passes through the backend's customize_spec
    (
        "vllm/v1/worker/gpu_model_runner.py",
        "                    spec = attn_module.get_attn_backend().customize_spec(spec)\n",
    ),
    # the MTP drafter layer is the same QSA owner (its cache is K8V4 too)
    ("vllm/models/qwen4_exp/nvidia/mtp.py", '                    layer_type="full_attention",\n'),
    # the side-cache ring (capacity 8 with 3 drafts) must divide the block size
    (
        "vllm/models/qwen4_exp/common/qsa_cache.py",
        "        capacity = self.compress_ratio * cdiv(span, self.compress_ratio)\n",
    ),
    # the prefix-match unit must divide every prefix-cacheable group's block size
    (
        "vllm/v1/core/kv_cache_utils.py",
        '            f"Invalid prefix_match_unit={hash_block_size}; prefix-cacheable "\n',
    ),
    # the warmup driver hands the owner's cache to the kernel-owned warmup
    (
        "vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py",
        "    attention_profiles = warmup_qsa_sparse_paged_attention(\n",
    ),
]

HEADER = (
    "# ARC3-KVQ-E083: K8V4 QSA main KV (fp8 e4m3 keys, nvfp4 values). nvfp4 bit helpers\n"
    "# and value quantiser from andreasgru/vllm-qwen38-flash-next-kv-quant (Apache-2.0,\n"
    "# Copyright andreasgru), after vllm-project/vllm#44389 and #54846, via the E062 port;\n"
    "# K8V4 row, writer and V-only read branch by ARC-AGI/experiments/e083_k8v4_port.py\n"
    "# for vLLM e9757321527c + ARC3 overlay e3a6fe0d.\n"
)

# ---------------------------------------------------------------------------
# Edits. ("replace", file, name, anchor, new) or ("span", file, name, start, end, new):
# a span replaces everything from the start anchor through the end anchor.
# ---------------------------------------------------------------------------

EDITS: list[tuple] = []

# ===== vllm/platforms/interface.py ==========================================
EDITS.append((
    "replace", PLAT, "hybrid-align-k8v4-page",
    """            attn_page_size_1_token = backend_cls.customize_spec(
                attn_spec
            ).page_size_bytes
""",
    """            attn_page_size_1_token = backend_cls.customize_spec(
                attn_spec
            ).page_size_bytes
            # ARC3-KVQ-E083: with ARC3_KVQ_K8V4=1 a QSA row is [K e4m3 D | V e2m1
            # D // 2 | V e4m3 scales D // 16]. The first non-SSM backend found may be
            # a QSA side cache that does not know this packing, so size the block
            # against the real row here (800 B/token/layer -> 2,128-token blocks).
            _arc3_k8v4 = os.environ.get("ARC3_KVQ_K8V4", "0").strip()
            if _arc3_k8v4 not in ("", "0", "1"):
                raise ValueError(f"ARC3_KVQ_K8V4 must be 0 or 1, got {_arc3_k8v4!r}")
            if _arc3_k8v4 == "1":
                if cache_config.cache_dtype not in ("fp8", "fp8_e4m3"):
                    raise ValueError(
                        "ARC3_KVQ_K8V4=1 needs --kv-cache-dtype fp8 or fp8_e4m3, "
                        f"got {cache_config.cache_dtype!r}"
                    )
                if "Qwen4Exp" not in str(model_config.architecture):
                    raise ValueError(
                        "ARC3_KVQ_K8V4=1 is only for Qwen4Exp QSA models, got "
                        f"{model_config.architecture!r}"
                    )
                _arc3_head = model_config.get_head_size()
                if _arc3_head % 16:
                    raise ValueError("ARC3_KVQ_K8V4 needs head_size % 16 == 0")
                attn_page_size_1_token = model_config.get_num_kv_heads(
                    parallel_config
                ) * (_arc3_head + _arc3_head // 2 + _arc3_head // 16)
""",
))

# ===== vllm/models/qwen4_exp/nvidia/qsa.py ==================================
EDITS.append((
    "replace", QSA, "imports-and-helpers",
    """from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer
""",
    """from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer

""" + HEADER + """import os as _k8v4_os  # ARC3-KVQ-E083
from dataclasses import replace as _k8v4_replace  # ARC3-KVQ-E083

from vllm.v1.kv_cache_interface import (  # ARC3-KVQ-E083
    KVQuantMode as _K8V4QuantMode,
)


def _k8v4_flag() -> bool:
    \"\"\"ARC3-KVQ-E083: ARC3_KVQ_K8V4=1 stores the QSA main KV as K8V4 (fp8 keys,
    nvfp4 values) when the cache dtype is fp8/fp8_e4m3. Unset or 0: stock fp8.\"\"\"
    raw = _k8v4_os.environ.get("ARC3_KVQ_K8V4", "0").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"ARC3_KVQ_K8V4 must be 0 or 1, got {raw!r}")
    return raw == "1"


# Read once per process at import (the server's environment is fixed at start).
_K8V4_ON = _k8v4_flag()


def _k8v4_row_bytes(head_size: int) -> int:
    \"\"\"ARC3-KVQ-E083: bytes of one (token, KV head) row: K e4m3 + V e2m1 + V scales.\"\"\"
    return head_size + head_size // 2 + head_size // 16


def _k8v4_spec(spec):
    \"\"\"ARC3-KVQ-E083: K8V4 page geometry, one cell of _k8v4_row_bytes per (KV head,
    token) in the existing head slots. Only fp8-per-tensor specs with the flag on;
    idempotent (a packed spec is returned unchanged).\"\"\"
    if (
        not _K8V4_ON
        or spec.state_content_bytes is not None
        or spec.kv_quant_mode != _K8V4QuantMode.FP8_PER_TENSOR
    ):
        return spec
    if spec.head_size != spec.head_size_v or spec.head_size % 16:
        raise NotImplementedError("K8V4 needs equal K/V head sizes, a multiple of 16")
    if spec.dtype != torch.uint8:
        raise NotImplementedError("K8V4 needs uint8 (fp8) cache storage")
    return _k8v4_replace(spec, state_content_bytes=_k8v4_row_bytes(spec.head_size))
""",
))
EDITS.append((
    "replace", QSA, "backend-customize-spec",
    """    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
""",
    """    @classmethod
    def customize_spec(cls, spec):
        # ARC3-KVQ-E083: the model runner passes every attention spec through this
        # hook; with the flag on, the fp8 page becomes the K8V4 page.
        return _k8v4_spec(super().customize_spec(spec))

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
""",
))
EDITS.append((
    "replace", QSA, "impl-flag",
    """        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
            )
        self.supports_quant_query_input = False
""",
    """        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
            )
        # ARC3-KVQ-E083: K8V4 is fp8 storage with another row (see _k8v4_spec).
        self.kv_cache_k8v4 = _K8V4_ON and self.kv_cache_dtype in ("fp8", "fp8_e4m3")
        self.supports_quant_query_input = False
""",
))
EDITS.append((
    "replace", QSA, "impl-writer",
    """    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
""",
    """    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # ARC3-KVQ-E083: the inherited writer splits a row at head_size into two
        # equal fp8 halves; a K8V4 row is [K e4m3 | V e2m1 | V scales].
        if not getattr(self, "kv_cache_k8v4", False):
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return None
        from .ops.qsa import k8v4_cache_views, k8v4_store

        k_rows, v_data, v_scales = k8v4_cache_views(kv_cache, self.head_size)
        k8v4_store(
            key,
            value,
            k_rows,
            v_data,
            v_scales,
            slot_mapping,
            layer._k_scale,
            layer._v_scale,
        )
        return None

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
""",
))
EDITS.append((
    "replace", QSA, "forward-views",
    """        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        k_scale = v_scale = None
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            # The cache is allocated as uint8; reinterpret the e4m3 bytes
            # (same itemsize, so shape and strides are preserved).
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
""",
    """        v_scale_cache = None  # ARC3-KVQ-E083
        if getattr(self, "kv_cache_k8v4", False):
            # ARC3-KVQ-E083: strided views of one row: K e4m3 bytes (viewed as e4m3
            # below, exactly as fp8), V packed e2m1 bytes, V block scales.
            from .ops.qsa import k8v4_cache_views

            key_cache, value_cache, v_scale_cache = k8v4_cache_views(
                kv_cache, self.head_size
            )
        else:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size, dim=-1
            )
        k_scale = v_scale = None
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            # The cache is allocated as uint8; reinterpret the e4m3 bytes
            # (same itemsize, so shape and strides are preserved).
            key_cache = key_cache.view(torch.float8_e4m3fn)
            if v_scale_cache is None:  # ARC3-KVQ-E083: K8V4 values stay packed
                value_cache = value_cache.view(torch.float8_e4m3fn)
""",
))
EDITS.append((
    "replace", QSA, "forward-kernel-call",
    """            k_scale=k_scale,
            v_scale=v_scale,
            output_gate=output_gate[:num_tokens],
        )
        return output
""",
    """            k_scale=k_scale,
            v_scale=v_scale,
            output_gate=output_gate[:num_tokens],
            v_scale_cache=v_scale_cache,  # ARC3-KVQ-E083
        )
        return output
""",
))
EDITS.append((
    "replace", QSA, "owner-gate",
    """            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
""",
    """            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"
            )
        if _K8V4_ON and cache_config.cache_dtype not in ("fp8", "fp8_e4m3"):
            # ARC3-KVQ-E083: the flag only re-packs an fp8 cache.
            raise NotImplementedError(
                "ARC3_KVQ_K8V4=1 needs --kv-cache-dtype fp8 or fp8_e4m3"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
""",
))
EDITS.append((
    "replace", QSA, "owner-spec",
    """    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
""",
    """    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # ARC3-KVQ-E083: the owner builds its own spec, so apply the K8V4 geometry
        # here as well as in customize_spec (both idempotent).
        spec = _k8v4_spec(
            FullAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                head_size_v=self.head_dim,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            )
        )
        padded = vllm_config.cache_config.mamba_page_size_padded
        if (
            spec.state_content_bytes is not None
            and padded is not None
            and padded != spec.page_size_bytes
        ):
            # The platform sized the hybrid block for another row (e.g. the flag
            # was not set in the process that ran the alignment): fail loudly.
            raise ValueError(
                f"ARC3-KVQ-E083: K8V4 page {spec.page_size_bytes} B != aligned "
                f"mamba page {padded} B; set ARC3_KVQ_K8V4=1 for every process"
            )
        return spec
""",
))

# ===== vllm/models/qwen4_exp/nvidia/ops/qsa.py ===============================
EDITS.append((
    "replace", OPS, "bit-helpers-and-kernel-pointer",
    """@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
""",
    HEADER + """# nvfp4 bit helpers (andreasgru kvq-patch.py, after vllm#44389; renamed from the
# E062 port). Architecture independent: no FP4 instructions, so no SM guard.
@triton.jit
def _k8v4_decode_e2m1(nibble):
    \"\"\"FP4 E2M1 nibble (sign + 3 magnitude bits) -> float32. For magnitude >= 2
    the value is 2^(exp - 1) * (1 + m / 2), i.e. the float32 bits
    (126 << 23) + (magnitude << 22); magnitudes 0 and 1 are 0.0 and 0.5.\"\"\"
    magnitude = nibble & 0x07
    magnitude_i32 = magnitude.to(tl.int32)
    sign_bits = ((nibble & 0x08).to(tl.uint32)) << 28
    normal_bits = ((126 << 23) + (magnitude_i32 << 22)).to(tl.uint32) | sign_bits
    normal = normal_bits.to(tl.uint32).to(tl.float32, bitcast=True)
    subnormal_bits = ((magnitude & 0x01).to(tl.uint32) * 0x3F000000) | sign_bits
    subnormal = subnormal_bits.to(tl.uint32).to(tl.float32, bitcast=True)
    return tl.where(magnitude < 2, subnormal, normal)


@triton.jit
def _k8v4_e4m3_to_float(bits):
    \"\"\"E4M3 block scale (magnitude only) -> float32. Bias 7 -> 127 is +120 on
    the exponent; subnormals are mant * 2^-9.\"\"\"
    payload = bits.to(tl.int32) & 0x7F
    exp_bits = (payload >> 3) & 0x0F
    mant = payload & 0x07
    normal_bits = ((exp_bits + 120) << 23) | (mant << 20)
    normal = normal_bits.to(tl.uint32).to(tl.float32, bitcast=True)
    subnormal = mant.to(tl.float32) / 512.0
    value = tl.where(exp_bits == 0, subnormal, normal)
    return tl.where(payload == 0, 0.0, value)


@triton.jit
def _k8v4_float_to_e4m3_bits(x):
    \"\"\"Non-negative float32 -> E4M3 bits, round to nearest even, saturating at
    448 (as __nv_cvt_float_to_fp8(..., __NV_SATFINITE, __NV_E4M3)).\"\"\"
    x = tl.minimum(tl.maximum(x, 0.0), 448.0)
    # Subnormals: mant * 2^-9 for x < 2^-6; mant 8 is the first normal, 0x08.
    r = x * 512.0
    f = tl.floor(r)
    d = r - f
    fi = f.to(tl.int32)
    sub = fi + tl.where((d > 0.5) | ((d == 0.5) & ((fi & 1) == 1)), 1, 0)
    sub = tl.minimum(sub, 8)
    x_bits = tl.maximum(x, 0.015625).to(tl.uint32, bitcast=True)
    exp_bits = ((x_bits >> 23) & 0xFF).to(tl.int32) - 120
    mant_full = (x_bits & 0x7FFFFF).to(tl.int32)
    rnd = ((mant_full >> 20) & 1) + 0x7FFFF
    mant = (mant_full + rnd) >> 20
    exp_bits += mant >> 3
    mant = mant & 7
    normal = (exp_bits << 3) | mant
    bits = tl.where(x < 0.015625, sub, normal)
    return bits.to(tl.uint8)


@triton.jit
def _k8v4_float_to_e2m1_bits(x):
    \"\"\"float32 -> FP4 E2M1 nibble, bounds as cvt.rn.satfinite.e2m1x2.f32.\"\"\"
    sign = tl.where(x < 0.0, 8, 0)
    abs_x = tl.abs(x)
    mag = tl.full(x.shape, 0, dtype=tl.int32)
    mag = tl.where((abs_x > 0.25) & (abs_x < 0.75), 1, mag)
    mag = tl.where((abs_x >= 0.75) & (abs_x <= 1.25), 2, mag)
    mag = tl.where((abs_x > 1.25) & (abs_x < 1.75), 3, mag)
    mag = tl.where((abs_x >= 1.75) & (abs_x <= 2.5), 4, mag)
    mag = tl.where((abs_x > 2.5) & (abs_x < 3.5), 5, mag)
    mag = tl.where((abs_x >= 3.5) & (abs_x <= 5.0), 6, mag)
    mag = tl.where(abs_x > 5.0, 7, mag)
    return (sign | mag).to(tl.uint8)


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    # ARC3-KVQ-E083: K8V4 value block-scale page (raw uint8 bits); only read when
    # V_NVFP4, otherwise any valid pointer.
    v_sf_ptr,
    indices_ptr,
""",
))
EDITS.append((
    "replace", OPS, "kernel-scale-strides",
    """    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
""",
    """    stride_v_block,
    stride_v_token,
    stride_v_head,
    # ARC3-KVQ-E083: value block-scale strides (the scale bytes sit in the same row
    # as K and V, so these equal the V data strides; passed explicitly anyway).
    stride_vs_block,
    stride_vs_token,
    stride_vs_head,
    stride_indices_row,
""",
))
EDITS.append((
    "replace", OPS, "kernel-constexpr",
    """    BLOCK_N: tl.constexpr,
    IS_FP8: tl.constexpr,
) -> None:
""",
    """    BLOCK_N: tl.constexpr,
    IS_FP8: tl.constexpr,
    V_NVFP4: tl.constexpr = False,  # ARC3-KVQ-E083
) -> None:
""",
))
EDITS.append((
    "replace", OPS, "kernel-k8v4-value-branch",
    """        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        if IS_FP8:
            # e4m3 -> Q dtype is exact; keep the QK dot in Q's dtype (fp8 QK
            # measured slower here and less accurate).
            keys = keys.to(query.dtype)
""",
    """        if V_NVFP4:
            # ARC3-KVQ-E083: K8V4 values. A row holds HEAD_DIM values in HEAD_DIM // 2
            # bytes (two E2M1 nibbles per byte, even index in the low nibble) and
            # HEAD_DIM // 16 E4M3 block scales right after them in the same row
            # (linear: group g of token t is byte g of t's scale run). value =
            # nibble * block scale; the layer scale is folded on the host into
            # output_scale, as for fp8. Scales are loaded at their natural width
            # [BLOCK_N, HEAD_DIM // 16] and broadcast in registers; the product is
            # exact in fp16 (<= 5 significant bits, 2^-10 .. 2688). Keys stay on
            # the fp8 path below.
            byte_offsets = dim_offsets // 2
            v_bytes = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + byte_offsets[None, :],
                mask=valid[:, None],
                other=0,
            )
            v_nib = tl.where(
                (dim_offsets[None, :] & 1) == 0, v_bytes & 0x0F, (v_bytes >> 4) & 0x0F
            )
            sf_groups = tl.arange(0, HEAD_DIM // 16)
            v_sf = _k8v4_e4m3_to_float(
                tl.load(
                    v_sf_ptr
                    + safe_page[:, None] * stride_vs_block
                    + page_offset[:, None] * stride_vs_token
                    + kv_head * stride_vs_head
                    + sf_groups[None, :],
                    mask=valid[:, None],
                    other=0,
                )
            )
            values = (
                _k8v4_decode_e2m1(v_nib)
                * tl.reshape(
                    tl.broadcast_to(v_sf[:, :, None], (BLOCK_N, HEAD_DIM // 16, 16)),
                    (BLOCK_N, HEAD_DIM),
                )
            ).to(tl.float16)
        else:
            values = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            )
        if IS_FP8:
            # e4m3 -> Q dtype is exact; keep the QK dot in Q's dtype (fp8 QK
            # measured slower here and less accurate).
            keys = keys.to(query.dtype)
""",
))
EDITS.append((
    "replace", OPS, "k8v4-config-helpers",
    """def _select_config(
    num_rows: int,
""",
    """def _k8v4_select_sm120_config(
    base_programs: int, use_prefill_config: bool
) -> tuple[int, int, int]:
    \"\"\"ARC3-KVQ-E083: (block_n, target_splits, num_warps) for K8V4 on sm_120.

    Not tuned for speed (no GPU yet). The E062 nvfp4 table: the same target_splits
    per bucket as the fp8 rows of _select_sm120_config (same program count and
    split-K merge cost), the tile capped at 32 columns and run with 8 warps, one
    stage (see _k8v4_num_stages). The V decode needs the registers and shared memory
    of E062's nvfp4 branch (sm_120 compile of that branch: ~1,024 B * BLOCK_N + 8 KB
    shared memory, 0-4 B spill at 32 columns / 8 warps); the fp8 rows' 64- and
    128-column decode tiles are a speed question for the GPU bench.
    \"\"\"
    if base_programs > 2048:
        return (32, 2, 8) if use_prefill_config else (32, 1, 8)
    if base_programs <= 24:
        return 32, 64, 8
    if base_programs <= 64:
        return 32, 8, 8
    if base_programs <= 512:
        return 32, 4, 8
    return 32, 2, 8


def _k8v4_num_stages(block_n: int, is_k8v4: bool) -> int:
    \"\"\"ARC3-KVQ-E083: pipeline stages; K8V4 wide tiles run one stage (E062 /
    andreasgru SM120 rule for the nvfp4 decode: two stages were slower).\"\"\"
    if is_k8v4 and block_n > 16:
        return 1
    return 2


def _select_config(
    num_rows: int,
""",
))
EDITS.append((
    "replace", OPS, "select-config-signature",
    """    is_fp8: bool = False,
) -> tuple[int, int, int, int]:
""",
    """    is_fp8: bool = False,
    is_k8v4: bool = False,  # ARC3-KVQ-E083
) -> tuple[int, int, int, int]:
""",
))
EDITS.append((
    "replace", OPS, "select-config-dispatch",
    """    base_programs = num_rows * num_kv_heads
    if _is_sm120():
        BLOCK_N, target_splits, num_warps = _select_sm120_config(
""",
    """    base_programs = num_rows * num_kv_heads
    if is_k8v4 and _is_sm120():  # ARC3-KVQ-E083
        BLOCK_N, target_splits, num_warps = _k8v4_select_sm120_config(
            base_programs, use_prefill_config
        )
    elif _is_sm120():
        BLOCK_N, target_splits, num_warps = _select_sm120_config(
""",
))
EDITS.append((
    "replace", OPS, "wrapper-signature-and-checks",
    """    *,
    output_gate: torch.Tensor,
) -> torch.Tensor:
    \"\"\"Run sparse GQA directly over paged BF16 or FP8-e4m3 K/V caches.
""",
    """    *,
    output_gate: torch.Tensor,
    v_scale_cache: torch.Tensor | None = None,  # ARC3-KVQ-E083
) -> torch.Tensor:
    \"\"\"Run sparse GQA directly over paged BF16 or FP8-e4m3 K/V caches.

    ARC3-KVQ-E083: K8V4 caches pass k_cache as the e4m3 key view (as fp8),
    v_cache as the packed e2m1 value bytes (uint8, last dim head_dim // 2) and
    v_scale_cache as the E4M3 value block scales (uint8, last dim head_dim // 16);
    k_scale/v_scale are folded exactly as for fp8.
""",
))
EDITS.append((
    "replace", OPS, "wrapper-shape-check",
    """    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
""",
    """    is_k8v4 = v_scale_cache is not None  # ARC3-KVQ-E083
    if (
        q.ndim != 3
        or k_cache.ndim != 4
        or v_cache.ndim != 4
        or (
            v_cache.shape[:3] != k_cache.shape[:3]
            if is_k8v4
            else v_cache.shape != k_cache.shape
        )
    ):
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
""",
))
EDITS.append((
    "replace", OPS, "wrapper-dtype-check",
    """    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == v_cache.dtype
    is_fp8 = k_cache.dtype == torch.float8_e4m3fn
""",
    """    assert q.dtype == torch.bfloat16
    if is_k8v4:  # ARC3-KVQ-E083
        _k8v4_check_pages(k_cache, v_cache, v_scale_cache, head_dim)
    else:
        assert k_cache.dtype == v_cache.dtype
    is_fp8 = k_cache.dtype == torch.float8_e4m3fn
""",
))
EDITS.append((
    "replace", OPS, "wrapper-select-config",
    """        q.shape[0], k_cache.shape[2], use_prefill_config, selection_width, is_fp8
    )
""",
    """        q.shape[0],
        k_cache.shape[2],
        use_prefill_config,
        selection_width,
        is_fp8,
        is_k8v4,  # ARC3-KVQ-E083
    )
""",
))
EDITS.append((
    "replace", OPS, "wrapper-launch",
    """    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
""",
    """    # ARC3-KVQ-E083: Triton needs a pointer for the compiled-out scale page too.
    if is_k8v4:
        v_sf_arg = v_scale_cache
        vs_strides = tuple(v_scale_cache.stride()[:3])
    else:
        v_sf_arg = q
        vs_strides = (0, 0, 0)

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        v_sf_arg,
        logical_indices,
""",
))
EDITS.append((
    "replace", OPS, "wrapper-launch-strides",
    """        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
""",
    """        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        vs_strides[0],
        vs_strides[1],
        vs_strides[2],
        logical_indices.stride(0),
""",
))
EDITS.append((
    "replace", OPS, "wrapper-launch-constexpr",
    """        IS_FP8=is_fp8,
        num_warps=partial_warps,
        num_stages=2,
    )
""",
    """        IS_FP8=is_fp8,
        V_NVFP4=is_k8v4,  # ARC3-KVQ-E083
        num_warps=partial_warps,
        num_stages=_k8v4_num_stages(block_n, is_k8v4),
    )
""",
))
EDITS.append((
    "span", OPS, "warmup-k8v4",
    """def warmup_qsa_sparse_paged_attention(
    kv_cache: torch.Tensor,
""",
    """        warmed.append((block_n, num_splits, warps))
    return tuple(warmed)
""",
    """def _k8v4_check_pages(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    head_dim: int,
) -> None:
    \"\"\"ARC3-KVQ-E083: shape/dtype/stride contract of the K8V4 views.\"\"\"
    if k_cache.dtype != torch.float8_e4m3fn:
        raise ValueError("QSA K8V4 keys must be an e4m3 view")
    if v_cache.dtype != torch.uint8 or v_scale_cache.dtype != torch.uint8:
        raise ValueError("QSA K8V4 value data and scales must be raw uint8 bytes")
    if v_cache.shape[3] != head_dim // 2:
        raise ValueError("QSA K8V4 value rows must be head_dim // 2 bytes")
    if (
        v_scale_cache.ndim != 4
        or v_scale_cache.shape[:3] != k_cache.shape[:3]
        or v_scale_cache.shape[3] != head_dim // 16
    ):
        raise ValueError("QSA K8V4 value block-scale page has the wrong shape")
    if v_scale_cache.stride(3) != 1 or v_cache.stride(3) != 1:
        raise ValueError("QSA K8V4 value rows must be contiguous")
    if v_scale_cache.device != k_cache.device or v_cache.device != k_cache.device:
        raise ValueError("QSA K8V4 pages must live on one device")


def _k8v4_is_cache(kv_cache: torch.Tensor) -> bool:
    \"\"\"ARC3-KVQ-E083: a K8V4 cache is uint8 [B, H, N, 25 * D / 16] (400 for D 256);
    fp8 caches are uint8 [B, H, N, 2 * D] (2 * D is never a multiple of 25 for a
    power-of-two D).\"\"\"
    full_dim = kv_cache.shape[-1]
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim != 4 or full_dim % 25:
        return False
    head_dim = full_dim * 16 // 25
    return head_dim >= 16 and (head_dim & (head_dim - 1)) == 0


def warmup_qsa_sparse_paged_attention(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_query_heads: int,
    selection_width: int,
) -> tuple[tuple[int, int, int], ...]:
    \"\"\"Compile every production-reachable split-K/merge specialization.

    ARC3-KVQ-E083: the stock version treats any uint8 cache as fp8 and splits it
    at half its row width; a K8V4 cache (rows of 400 bytes) would warm wrong
    shapes (HEAD_DIM 200, a non-power-of-two arange) and fail at startup.
    \"\"\"
    is_k8v4 = _k8v4_is_cache(kv_cache)
    if is_k8v4:
        head_dim = kv_cache.shape[-1] * 16 // 25
        key_cache, value_cache, v_sf = k8v4_cache_views(kv_cache, head_dim)
        is_fp8 = True
        k_dtype, v_dtype = torch.float8_e4m3fn, torch.uint8
    else:
        head_dim = kv_cache.shape[-1] // 2
        key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
        v_sf = None
        # An fp8 cache is allocated as uint8 and viewed as e4m3 at attention time.
        is_fp8 = kv_cache.dtype == torch.uint8
        k_dtype = v_dtype = torch.float8_e4m3fn if is_fp8 else key_cache.dtype
    num_kv_heads = key_cache.shape[2]
    group_size = num_query_heads // num_kv_heads
    block_m = triton.next_power_of_2(group_size)

    # Every config the dispatch can pick for this group size.
    profiles = {
        _select_config(
            num_rows,
            num_kv_heads,
            use_prefill_config,
            selection_width,
            is_fp8,
            is_k8v4,
        )
        for num_rows in range(1, 8193)
        for use_prefill_config in (False, True)
    }

    # Scalars constant per deployment get their real values (their divisibility
    # specialization is wanted); the batch-varying ones are do_not_specialize'd
    # on the kernels, so any value here compiles the only variant.
    num_rows = 16
    num_requests = 16
    q_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    k_cache_ptr = TritonWarmupTensor(
        k_dtype,
        shape=tuple(key_cache.shape),
        strides=tuple(key_cache.stride()),
    )
    v_cache_ptr = TritonWarmupTensor(
        v_dtype,
        shape=tuple(value_cache.shape),
        strides=tuple(value_cache.stride()),
    )
    if is_k8v4:
        v_sf_ptr = TritonWarmupTensor(
            torch.uint8, shape=tuple(v_sf.shape), strides=tuple(v_sf.stride())
        )
        vs_strides = tuple(v_sf.stride()[:3])
    else:
        # The runtime passes q for the compiled-out scale page.
        v_sf_ptr = q_ptr
        vs_strides = (0, 0, 0)
    # +1: the packed buffer's trailing count column.
    indices_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, selection_width + 1))
    block_table_ptr = TritonWarmupTensor(
        block_table.dtype,
        shape=tuple(block_table.shape),
        strides=tuple(block_table.stride()),
    )
    token_to_req_ptr = TritonWarmupTensor(torch.int32)
    output_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    # The output gate is mandatory at runtime; warm the gated specialization.
    output_gate_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    head_stride = head_dim
    row_stride = num_query_heads * head_dim
    num_cache_blocks = triton_scalar_specialization_rep(kv_cache.shape[0])

    warmed = []
    for block_n, warps, num_tiles, num_splits in sorted(profiles):
        if num_splits == 1:
            partial_output_ptr = output_ptr
            partial_lse_ptr = output_ptr
        else:
            partial_output_ptr = TritonWarmupTensor(
                torch.float32,
                shape=(num_splits, num_rows, num_query_heads, head_dim),
            )
            partial_lse_ptr = TritonWarmupTensor(
                torch.float32, shape=(num_splits, num_rows, num_query_heads)
            )
        _qsa_sparse_paged_gqa_splitk_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            v_sf_ptr,
            indices_ptr,
            block_table_ptr,
            token_to_req_ptr,
            partial_output_ptr,
            partial_lse_ptr,
            output_ptr,
            1.0,
            1.0,
            output_gate_ptr,
            row_stride,
            head_stride,
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            vs_strides[0],
            vs_strides[1],
            vs_strides[2],
            selection_width + 1,
            block_table.stride(0),
            row_stride,
            head_stride,
            row_stride,
            head_stride,
            num_rows,
            num_cache_blocks,
            num_requests,
            TOPK=selection_width,
            PAGE_SIZE=key_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            NUM_SPLITS=num_splits,
            NUM_TILES=num_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            IS_FP8=is_fp8,
            V_NVFP4=is_k8v4,
            num_warps=warps,
            num_stages=_k8v4_num_stages(block_n, is_k8v4),
            grid=(num_rows, num_kv_heads, num_splits),
        )
        if num_splits > 1:
            _qsa_merge_splitk_kernel.warmup(
                partial_output_ptr,
                partial_lse_ptr,
                output_ptr,
                output_gate_ptr,
                row_stride,
                head_stride,
                row_stride,
                head_stride,
                num_rows,
                HEAD_DIM=head_dim,
                NUM_QUERY_HEADS=num_query_heads,
                NUM_SPLITS=num_splits,
                BLOCK_SPLITS=triton.next_power_of_2(num_splits),
                num_warps=2,
                num_stages=1,
                grid=(num_rows, num_query_heads),
            )
        warmed.append((block_n, num_splits, warps))
    return tuple(warmed)
""",
))
EDITS.append((
    "replace", OPS, "k8v4-views-and-writer",
    """def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
""",
    """def k8v4_cache_views(
    kv_cache: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    \"\"\"ARC3-KVQ-E083: split a K8V4 QSA cache [B, H, N, D + D // 2 + D // 16] (uint8)
    into (k_e4m3_bytes [B, N, H, D], v_data [B, N, H, D // 2], v_scales
    [B, N, H, D // 16]), strided uint8 views of the same rows, no copies. Every
    layout keeps a row's bytes contiguous, so no layout is required.\"\"\"
    if kv_cache.dtype != torch.uint8 or kv_cache.ndim != 4:
        raise ValueError("QSA K8V4 cache must be a 4D uint8 tensor")
    if head_dim % 16 or kv_cache.shape[3] != head_dim + head_dim // 2 + head_dim // 16:
        raise ValueError("QSA K8V4 cache rows must be D + D // 2 + D // 16 bytes")
    if kv_cache.stride(3) != 1:
        raise ValueError("QSA K8V4 cache rows must be dense")
    rows = kv_cache.transpose(1, 2)
    half = head_dim // 2
    return (
        rows[..., :head_dim],
        rows[..., head_dim : head_dim + half],
        rows[..., head_dim + half :],
    )


@triton.jit
def _k8v4_store_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_data_ptr,
    v_sf_ptr,
    slot_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_key_token,
    stride_key_head,
    stride_val_token,
    stride_val_head,
    s_k_block,
    s_k_token,
    s_k_head,
    s_vd_block,
    s_vd_token,
    s_vd_head,
    s_vs_block,
    s_vs_token,
    s_vs_head,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
) -> None:
    \"\"\"ARC3-KVQ-E083: K8V4 writer, one program per (token, KV head).

    Keys: e4m3 of key / k_scale, round-to-nearest-even, saturating (the bytes
    reshape_and_cache_flash writes for fp8). Values: nvfp4 as the E062 / andreasgru
    software writer (block scale = layer-scaled amax / 6 in E4M3, values in E2M1),
    scales linear in the row. A slot < 0 (CUDA-graph padding) writes nothing.
    \"\"\"
    tl.static_assert(HEAD_DIM % 16 == 0)
    GROUPS: tl.constexpr = HEAD_DIM // 16

    token = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_ptr + token).to(tl.int64)
    valid = slot >= 0
    safe_slot = tl.maximum(slot, 0)
    block = safe_slot // PAGE_SIZE
    offset = safe_slot % PAGE_SIZE

    # Keys -> e4m3 bytes (sign from the float bits, so -0.0 keeps its sign).
    dims = tl.arange(0, HEAD_DIM)
    k = tl.load(key_ptr + token * stride_key_token + head * stride_key_head + dims)
    k = k.to(tl.float32)
    k_global = tl.load(k_scale_ptr).to(tl.float32)
    k_div = tl.where(k_global == 0.0, 1.0, k_global)
    kq = tl.div_rn(k, tl.zeros_like(k) + k_div)
    k_sign = ((kq.to(tl.uint32, bitcast=True) >> 31) << 7).to(tl.uint8)
    k_bits = _k8v4_float_to_e4m3_bits(tl.abs(kq)) | k_sign
    tl.store(
        k_cache_ptr + block * s_k_block + offset * s_k_token + head * s_k_head + dims,
        k_bits,
        mask=(dims < HEAD_DIM) & valid,
    )

    # Values -> nvfp4: [GROUPS, 8] bytes, even values in the low nibble.
    groups = tl.arange(0, GROUPS)
    pairs = tl.arange(0, 8)
    v_row = value_ptr + token * stride_val_token + head * stride_val_head
    v_even = tl.load(v_row + groups[:, None] * 16 + pairs[None, :] * 2).to(tl.float32)
    v_odd = tl.load(v_row + groups[:, None] * 16 + pairs[None, :] * 2 + 1).to(tl.float32)
    # The layer scale acts as the global scale: the writer divides by it, the
    # reader multiplies by it again (1.0 here).
    v_global = tl.load(v_scale_ptr).to(tl.float32)
    v_quant = tl.where(v_global == 0.0, 1.0, 1.0 / v_global)
    v_max = tl.maximum(tl.max(tl.abs(v_even), axis=1), tl.max(tl.abs(v_odd), axis=1))
    v_sf_bits = _k8v4_float_to_e4m3_bits((v_quant * v_max) / 6.0)
    v_sf_f = _k8v4_e4m3_to_float(v_sf_bits)
    v_out = tl.where(v_sf_f == 0.0, 0.0, v_quant / v_sf_f)
    v_low = _k8v4_float_to_e2m1_bits(
        tl.minimum(tl.maximum(v_even * v_out[:, None], -6.0), 6.0)
    )
    v_high = _k8v4_float_to_e2m1_bits(
        tl.minimum(tl.maximum(v_odd * v_out[:, None], -6.0), 6.0)
    )
    pair_mask = (pairs[None, :] < 8) & valid
    tl.store(
        v_data_ptr
        + block * s_vd_block
        + offset * s_vd_token
        + head * s_vd_head
        + groups[:, None] * 8
        + pairs[None, :],
        v_low | (v_high << 4),
        mask=pair_mask,
    )
    tl.store(
        v_sf_ptr + block * s_vs_block + offset * s_vs_token + head * s_vs_head + groups,
        v_sf_bits,
        mask=(groups < GROUPS) & valid,
    )


def k8v4_store(
    key: torch.Tensor,
    value: torch.Tensor,
    k_rows: torch.Tensor,
    v_data: torch.Tensor,
    v_sf: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    \"\"\"ARC3-KVQ-E083: write K/V [tokens, heads, head_dim] (bf16) into K8V4 rows
    (views from k8v4_cache_views). The token count is slot_mapping's length (K/V may
    be padded longer, as for reshape_and_cache_flash).\"\"\"
    if not HAS_TRITON:
        raise RuntimeError("QSA K8V4 store requires Triton")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("QSA K8V4 store expects [tokens, heads, head_dim] K/V")
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise ValueError("QSA K8V4 store expects BF16 inputs")
    head_dim = key.shape[2]
    if head_dim % 16 or key.stride(2) != 1 or value.stride(2) != 1:
        raise ValueError("QSA K8V4 store requires contiguous head_dim % 16 == 0")
    if (
        k_rows.dtype != torch.uint8
        or v_data.dtype != torch.uint8
        or v_sf.dtype != torch.uint8
    ):
        raise ValueError("QSA K8V4 store expects uint8 cache views")
    if (
        k_rows.shape[3] != head_dim
        or v_data.shape[3] != head_dim // 2
        or v_sf.shape[3] != head_dim // 16
    ):
        raise ValueError("QSA K8V4 store: cache row widths do not match head_dim")
    if k_rows.stride(3) != 1 or v_data.stride(3) != 1 or v_sf.stride(3) != 1:
        raise ValueError("QSA K8V4 store: cache rows must be contiguous")
    if k_rows.shape[2] != key.shape[1]:
        raise ValueError("QSA K8V4 store: head count does not match the cache")
    num_tokens = slot_mapping.numel()
    if num_tokens == 0:
        return
    if key.shape[0] < num_tokens:
        raise ValueError("QSA K8V4 store: fewer K/V rows than slots")
    _k8v4_store_kernel[(num_tokens, key.shape[1])](
        key,
        value,
        k_rows,
        v_data,
        v_sf,
        slot_mapping,
        k_scale,
        v_scale,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        k_rows.stride(0),
        k_rows.stride(1),
        k_rows.stride(2),
        v_data.stride(0),
        v_data.stride(1),
        v_data.stride(2),
        v_sf.stride(0),
        v_sf.stride(1),
        v_sf.stride(2),
        PAGE_SIZE=k_rows.shape[1],
        HEAD_DIM=head_dim,
        num_warps=2,
        num_stages=1,
    )


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
""",
))


# ---------------------------------------------------------------------------
# Engine (same contract as E062)
# ---------------------------------------------------------------------------

# sha256 of the three targets after this exact port (LF). analysis/test_e083_k8v4.py
# fails if a fresh apply of EDITS to the pinned originals produces anything else, so
# "already applied" can be recognised without shipping the originals.
PATCHED_PINS = {
    QSA: "cd9004607e740500cec8b8a30a8cbc005f6ab1b1f2e5134f9c28c40ede8c83a3",
    OPS: "b0fffccae898247a9aa0a55e2caa4bfcae6408db849aac4bd7a98a3b0fa35bab",
    PLAT: "25e65942f4261287cd264718eb1a122b94b6f8aac5dab29d1f29a877591d5ec4",
}


class Refused(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def resolve_site(arg: str | None) -> Path:
    raw = arg or os.environ.get("ARC3_KVQ_SITE")
    if not raw:
        raise Refused(2, "no site given (--site or ARC3_KVQ_SITE)")
    site = Path(raw)
    if (site / "vllm" / "__init__.py").is_file():
        return site
    if site.name == "vllm" and (site / "__init__.py").is_file():
        return site.parent
    raise Refused(2, f"no vllm package under {site}")


def apply_edits(text: str, edits: list[tuple], label: str) -> str:
    """Apply one file's edits in memory; every anchor must match exactly once."""
    for edit in edits:
        kind, name = edit[0], edit[2]
        if kind == "replace":
            anchor, new = edit[3], edit[4]
            n = text.count(anchor)
            if n != 1:
                raise Refused(3, f"{label}/{name}: anchor found {n}x (expected 1x)")
            text = text.replace(anchor, new, 1)
        elif kind == "span":
            start, end, new = edit[3], edit[4], edit[5]
            ns, ne = text.count(start), text.count(end)
            if ns != 1 or ne != 1:
                raise Refused(
                    3, f"{label}/{name}: span anchors found {ns}x/{ne}x (expected 1x/1x)"
                )
            i, j = text.index(start), text.index(end)
            if j < i:
                raise Refused(3, f"{label}/{name}: span end precedes its start")
            text = text[:i] + new + text[j + len(end) :]
        else:  # pragma: no cover - table error
            raise Refused(3, f"{label}/{name}: unknown edit kind {kind!r}")
    return text


def edits_by_file() -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {rel: [] for rel in TARGET_PINS}
    for edit in EDITS:
        out[edit[1]].append(edit)
    return out


def check_pins(site: Path, overlay_identity: Path | None) -> None:
    """Refuse unless the installed runtime is vLLM e975732 + overlay e3a6fe0d."""
    version_file = site / "vllm" / "_version.py"
    if version_file.is_file():
        text = version_file.read_text(encoding="utf-8", errors="replace")
        if VLLM_VERSION_TAG not in text and VLLM_COMMIT not in text:
            raise Refused(2, f"vllm/_version.py is not commit {VLLM_COMMIT[:12]}")
    if overlay_identity is not None:
        ident = json.loads(Path(overlay_identity).read_text(encoding="utf-8"))
        if ident.get("overlay_sha256") != OVERLAY_SHA256:
            raise Refused(2, f"overlay is {ident.get('overlay_sha256')}, pinned {OVERLAY_SHA256[:8]}")
        if ident.get("vllm_commit") != VLLM_COMMIT:
            raise Refused(2, f"overlay targets vLLM {ident.get('vllm_commit')}")
    for rel, pin in DEPENDENCY_PINS.items():
        path = site / rel
        if not path.is_file():
            raise Refused(2, f"missing {rel}")
        if _sha(path.read_bytes()) != pin:
            raise Refused(2, f"{rel} differs from vLLM e975732 + overlay e3a6fe0d")
    for rel, fact in DEPENDENCY_FACTS:
        n = _read(site / rel).count(fact)
        if n != 1:
            raise Refused(2, f"{rel}: expected text found {n}x: {fact.strip()[:60]}")


def plan(site: Path) -> tuple[str, dict[str, tuple[str, str | None]]]:
    """Classify every target without writing: returns (state, {rel: (current, new)}).

    A target is "original" (pinned stock sha; new = patched content, built and
    compiled in memory) or "patched" (sha equals PATCHED_PINS; new = None). Any
    other content is refused (including E062's nvfp4 port of the same files).
    state: "apply" (all original), "applied" (all patched) or "partial" (a mix,
    e.g. after an interrupted earlier run).
    """
    by_file = edits_by_file()
    files: dict[str, tuple[str, str | None]] = {}
    states: dict[str, str] = {}
    for rel, pin in TARGET_PINS.items():
        path = site / rel
        if not path.is_file():
            raise Refused(2, f"missing {rel}")
        current = _read(path)
        current_sha = _sha(current.encode("utf-8"))
        if current_sha == pin:
            new = apply_edits(current, by_file[rel], rel)
            try:
                compile(new, str(path), "exec")
            except SyntaxError as exc:
                raise Refused(3, f"{rel}: patched source does not compile: {exc}") from exc
            if PATCHED_PINS[rel] and _sha(new.encode("utf-8")) != PATCHED_PINS[rel]:
                raise Refused(3, f"{rel}: patched content differs from PATCHED_PINS")
            files[rel], states[rel] = (current, new), "original"
        elif PATCHED_PINS[rel] and current_sha == PATCHED_PINS[rel]:
            files[rel], states[rel] = (current, None), "patched"
        elif MARKER in current:
            raise Refused(2, f"{rel} carries {MARKER} but is not this version of the port")
        elif OTHER_PORT_MARKER in current:
            raise Refused(2, f"{rel} carries the E062 nvfp4 port; the two ports exclude each other")
        else:
            raise Refused(2, f"{rel} differs from the pinned vLLM e975732 file")
    if all(s == "patched" for s in states.values()):
        return "applied", files
    if all(s == "original" for s in states.values()):
        return "apply", files
    return "partial", files


def _remove_stale_bytecode(path: Path) -> int:
    removed = 0
    cache = path.parent / "__pycache__"
    if cache.is_dir():
        for pyc in cache.glob(f"{path.stem}.*.pyc"):
            try:
                pyc.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def write_all(site: Path, files: dict[str, tuple[str, str | None]]) -> tuple[int, int]:
    """Write every target that has new content; on any failure restore the files
    already replaced (their old content is still in memory) and refuse."""
    done: list[tuple[Path, str]] = []
    try:
        for rel, (current, new) in files.items():
            if new is None:
                continue
            path = site / rel
            tmp = path.with_name(f".{path.name}.e083.tmp")
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(new)
            os.replace(tmp, path)
            done.append((path, current))
            if _sha(path.read_bytes()) != _sha(new.encode("utf-8")):
                raise OSError(f"{rel}: content after write does not match")
    except Exception as exc:
        for path, current in reversed(done):
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(current)
        raise Refused(4, f"write failed and was rolled back: {exc}") from exc
    removed = sum(_remove_stale_bytecode(path) for path, _ in done)
    return len(done), removed


def run(site_arg: str | None, overlay_identity: str | None, check_only: bool) -> int:
    n_edits = len(EDITS)
    pin = f"vLLM {VLLM_COMMIT[:12]} + overlay {OVERLAY_SHA256[:8]}"
    try:
        site = resolve_site(site_arg)
        check_pins(site, Path(overlay_identity) if overlay_identity else None)
        state, files = plan(site)
        if state == "applied":
            print(
                f"[e083-k8v4] already applied: K8V4 QSA KV port present in "
                f"{len(TARGET_PINS)} files ({n_edits} edits), {pin}; nothing changed"
            )
            return 0
        if check_only:
            print(f"[e083-k8v4] check ok ({state}): pins and {n_edits} anchors verified; nothing written")
            return 0
        written, removed = write_all(site, files)
        print(
            f"[e083-k8v4] applied ({state}): K8V4 QSA KV port, {written} files / "
            f"{n_edits} edits, {pin}, {removed} stale .pyc removed; nvfp4 value helpers "
            "from andreasgru/vllm-qwen38-flash-next-kv-quant (Apache-2.0); "
            f"inert unless {FLAG_ENV}=1 with --kv-cache-dtype fp8_e4m3"
        )
        return 0
    except Refused as exc:
        print(f"[e083-k8v4] REFUSED, nothing changed: {exc}", file=sys.stderr)
        return exc.code


# ---------------------------------------------------------------------------
# Geometry (pure arithmetic; the same numbers the patched alignment produces).
# ---------------------------------------------------------------------------

HEAD_DIM = 256
NUM_KV_HEADS = 2
GDN_CONV_BYTES_PER_DRAFT_SLOT = 10240 * 2      # conv width x bf16
GDN_SSM_BYTES = 48 * 128 * 128 * 2             # SSM state, bf16
NUM_QSA_LAYERS = 13                            # 12 main + 1 MTP
INDEXER_BYTES_PER_STATE = 128                  # compressed indexer key per 4 tokens
COMPRESS_RATIO = 4


def gdn_state_bytes(num_drafts: int) -> int:
    """GDN state per linear-attention layer: conv (kernel 4 - 1 + drafts) + SSM."""
    return GDN_CONV_BYTES_PER_DRAFT_SLOT * (4 - 1 + num_drafts) + GDN_SSM_BYTES


def row_bytes_per_token(kind: str, head_dim: int = HEAD_DIM, heads: int = NUM_KV_HEADS) -> int:
    per_head = {
        "fp8": 2 * head_dim,
        "k8v4": head_dim + head_dim // 2 + head_dim // 16,
        "nvfp4": 2 * (head_dim // 2 + head_dim // 16),
    }[kind]
    return heads * per_head


def hybrid_block_tokens(bytes_per_token: int, num_drafts: int = 3, align: int = 16) -> int:
    """16 * ceil(GDN state / (16 * KV bytes per token per layer)), as
    Platform._align_hybrid_block_size computes it for this model."""
    state = gdn_state_bytes(num_drafts)
    return align * -(-state // (align * bytes_per_token))


def block_bytes(kind: str, num_drafts: int = 3) -> int:
    """Pool bytes per block: 13 QSA layers x (attention page + indexer page)."""
    bpt = row_bytes_per_token(kind)
    bs = hybrid_block_tokens(bpt, num_drafts)
    return NUM_QSA_LAYERS * (bs * bpt + bs // COMPRESS_RATIO * INDEXER_BYTES_PER_STATE)


def prefix_units(block: int, ring: int = 8, lo: int = 96, hi: int = 160) -> list[int]:
    """Prefix-match units in [lo, hi] that divide the block, are multiples of the
    compression ratio (indexer tokens_per_state) and of nothing else required."""
    return [u for u in range(lo, hi + 1) if block % u == 0 and u % COMPRESS_RATIO == 0]


# ---------------------------------------------------------------------------
# Kernel self-test. On the GPU it runs in the server's environment before the
# server starts (--selftest); on a CPU it runs under TRITON_INTERPRET=1
# (analysis/test_e083_k8v4.py). It needs no model and ~30 MB.
# ---------------------------------------------------------------------------

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_k8v4_rows(k_rows, v_data, v_sf, blocks, offsets):
    """Independent torch decode of written rows -> (K float32 [T, H, D], V float32
    [T, H, D]). Views as from k8v4_cache_views; blocks, offsets: int64 [T]."""
    import torch

    k = k_rows.cpu()[blocks, offsets].contiguous().view(torch.float8_e4m3fn).float()
    table = torch.tensor(E2M1_VALUES + tuple(-v for v in E2M1_VALUES))
    rows = v_data.cpu()[blocks, offsets].long()  # [T, H, D // 2]
    num_tokens, num_heads, half = rows.shape
    nib = torch.stack([rows & 0x0F, rows >> 4], dim=-1).reshape(num_tokens, num_heads, 2 * half)
    scale = v_sf.cpu()[blocks, offsets].contiguous().view(torch.float8_e4m3fn).float()
    v = table[nib] * scale.repeat_interleave(16, dim=-1)
    return k, v


def reference_e4m3(x, scale: float = 1.0):
    """What reshape_and_cache_flash stores for fp8: e4m3(x / scale), RNE, satfinite."""
    import torch

    return (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def selftest(
    ops,
    views_fn,
    layout_cls,
    device: str = "cuda",
    native_writer=None,
    page_size: int = 2128,
    num_blocks: int = 6,
    num_layers: int = 2,
    head_dim: int = 256,
    n_kv: int = 2,
    n_q: int = 24,
    width: int = 64,
    num_tokens: int = 96,
    profiles: tuple[tuple[int, bool], ...] = ((16, False), (72, False), (144, False), (300, True)),
    layouts: tuple[str, ...] = ("BLNHC", "BLHNC"),
    log=print,
) -> dict:
    """Check the K8V4 path at the production page geometry; returns a verdict dict.

    ops: the patched ops/qsa.py module; views_fn: create_kv_cache_views (the
    allocator's own view code); layout_cls: KVCacheLayout; native_writer:
    reshape_and_cache_flash or None. Checks, per layout: writer (keys bit-exact to
    the fp8 reference quantiser and, on the GPU, to the native fp8 writer; values at
    nvfp4 noise level, and a wrong scale lookup clearly worse), isolate (no byte
    outside the written rows changes, the other layer untouched), kernel (decode and
    prefill profiles vs a float64 reference on the dequantised values, and vs the
    unchanged bf16 branch of the same kernel on the same values).
    """
    import math
    from types import SimpleNamespace

    import torch

    full = head_dim + head_dim // 2 + head_dim // 16
    gen = torch.Generator().manual_seed(1234)
    verdict: dict = {"ok": False, "checks": {}, "per_layout": {}}
    spec = SimpleNamespace(
        num_heads=n_kv,
        block_size=page_size,
        state_content_size_bytes=full,
        page_size_padded=None,
        dtype=torch.uint8,
        get_num_kernel_states=lambda bs: bs,
    )
    page = n_kv * page_size * full

    def make_cache(layout_name: str):
        raw = torch.zeros(num_layers * num_blocks * page, dtype=torch.int8, device=device)
        tensor = SimpleNamespace(
            layers=[f"layer{i}" for i in range(num_layers)],
            layer_stride=page,
            block_stride=num_layers * page,
            offset=0,
        )
        views = views_fn(raw, spec, num_blocks, getattr(layout_cls, layout_name), tensor)
        return raw, views

    def rel(a, b):
        return float((a - b).norm() / b.norm())

    one = torch.ones(1, dtype=torch.float32, device=device)
    # Distinct slots, always including both ends of a block and of the cache.
    fixed = torch.tensor([0, page_size - 1, page_size, num_blocks * page_size - 1])
    rest = torch.randperm(num_blocks * page_size, generator=gen)
    slots = torch.cat([fixed, rest[~torch.isin(rest, fixed)][: num_tokens - len(fixed)]])
    blocks, offsets = slots // page_size, slots % page_size
    # Every 16-group gets its own magnitude, so a wrong scale lookup shows.
    groups = head_dim // 16
    amp = torch.logspace(-2, 1, groups)[torch.randperm(groups, generator=gen)]
    key = (torch.randn(num_tokens, n_kv, head_dim, generator=gen) * amp.repeat_interleave(16) * 3).to(torch.bfloat16)
    value = (torch.randn(num_tokens, n_kv, head_dim, generator=gen) * amp.flip(0).repeat_interleave(16)).to(torch.bfloat16)
    key_d, value_d = key.to(device), value.to(device)
    slots_d = slots.to(device=device, dtype=torch.int64)
    ref_k = reference_e4m3(key)

    ok_all = True
    worst, worst_twin = 0.0, 0.0
    for layout_name in layouts:
        lv: dict = {}
        raw, caches = make_cache(layout_name)
        cache = caches[1]
        k_rows, v_data, v_sf = ops.k8v4_cache_views(cache, head_dim)
        ops.k8v4_store(key_d, value_d, k_rows, v_data, v_sf, slots_d, one, one)

        # writer: keys
        dk, dv = decode_k8v4_rows(k_rows, v_data, v_sf, blocks, offsets)
        got_k_bits = k_rows.cpu()[blocks, offsets]
        lv["k_bitexact_vs_reference"] = bool(torch.equal(got_k_bits, ref_k.view(torch.uint8)))
        if native_writer is not None:
            # fp8 twin cache [B, N, H, 2D] written by the stock CUDA writer
            twin_raw = torch.zeros(num_blocks, page_size, n_kv, 2 * head_dim, dtype=torch.uint8, device=device)
            twin_k, twin_v = twin_raw.split(head_dim, dim=-1)
            native_writer(key_d, value_d, twin_k, twin_v, slots_d, "fp8", one, one)
            lv["k_bitexact_vs_native_fp8_writer"] = bool(
                torch.equal(got_k_bits, twin_k.cpu()[blocks, offsets])
            )
        # writer: values (round trip and a wrong-lookup contrast)
        v_rel = rel(dv, value.float())
        wrong = dv.clone()
        sf = v_sf.cpu()[blocks, offsets].contiguous().view(torch.float8_e4m3fn).float()
        wrong_scale = sf.roll(1, dims=-1).repeat_interleave(16, dim=-1)
        wrong = (dv / sf.repeat_interleave(16, dim=-1).clamp_min(1e-30)) * wrong_scale
        v_rel_wrong = rel(wrong, value.float())
        lv["v_roundtrip_rel"] = round(v_rel, 4)
        lv["v_wrong_scale_rel"] = round(v_rel_wrong, 4)
        lv["k_roundtrip_rel"] = round(rel(dk, key.float()), 4)
        writer_ok = (
            lv["k_bitexact_vs_reference"]
            and lv.get("k_bitexact_vs_native_fp8_writer", True)
            and 0.03 < v_rel < 0.15
            and v_rel_wrong > 3 * v_rel
        )
        lv["writer"] = bool(writer_ok)

        # isolate: only the written rows may be non-zero, the other layer untouched
        flat = raw.view(torch.uint8).cpu()
        touched = torch.zeros(flat.numel(), dtype=torch.bool)
        base_ptr = raw.data_ptr()
        start = cache.data_ptr() - base_ptr
        for b, t in zip(blocks.tolist(), offsets.tolist()):
            for h in range(n_kv):
                o = start + b * cache.stride(0) + h * cache.stride(1) + t * cache.stride(2)
                touched[o : o + full] = True
        stray = int((flat[~touched] != 0).sum())
        lv["stray_nonzero_bytes"] = stray
        lv["isolate"] = stray == 0 and bool((caches[0] == 0).all())

        # kernel vs float64 reference on the dequantised values
        slot_row = {s: i for i, s in enumerate(slots.tolist())}
        n_req = 2
        block_table = torch.stack([torch.randperm(num_blocks, generator=gen) for _ in range(n_req)]).to(torch.int32)
        logical = torch.empty(n_req, num_tokens, dtype=torch.int32)
        for r in range(n_req):
            where = {int(p): lb for lb, p in enumerate(block_table[r].tolist())}
            logical[r] = torch.tensor([where[int(b)] * page_size + int(t) for b, t in zip(blocks, offsets)])
        # bf16 twin: the same dequantised values (exact in bf16) in a bf16 cache,
        # read by the unchanged bf16 branch of the same kernel
        twin = torch.zeros(num_blocks, n_kv, page_size, 2 * head_dim, dtype=torch.bfloat16)
        twin[blocks, :, offsets, :head_dim] = dk.to(torch.bfloat16)
        twin[blocks, :, offsets, head_dim:] = dv.to(torch.bfloat16)
        twin = twin.to(device)
        twin_k, twin_v = twin.transpose(1, 2).split(head_dim, dim=-1)
        k_e4m3 = k_rows.view(torch.float8_e4m3fn)
        ok_kernel = True
        for rows, prefill in profiles:
            w = min(width, num_tokens)
            indices = torch.full((rows, w + 1), -1, dtype=torch.int32)
            token_to_req = torch.arange(rows, dtype=torch.int32) % n_req
            for r in range(rows):
                count = w - (r % 5)
                pick = torch.randperm(num_tokens, generator=gen)[:count]
                indices[r, :count] = logical[int(token_to_req[r]), pick]
                indices[r, w] = count
            q = torch.randn(rows, n_q, head_dim, generator=gen).to(torch.bfloat16)
            gate = torch.randn(rows, n_q, head_dim, generator=gen).to(torch.bfloat16)
            out = torch.empty_like(q, device=device)
            ops.qsa_sparse_paged_attention(
                q.to(device), k_e4m3, v_data, indices.to(device), block_table.to(device),
                token_to_req.to(device), prefill, out, 1.0, 1.0, output_gate=gate.to(device),
                v_scale_cache=v_sf,
            )
            got = out.double().cpu()
            ref = torch.empty(rows, n_q, head_dim, dtype=torch.float64)
            for r in range(rows):
                req, count = int(token_to_req[r]), int(indices[r, w])
                phys = []
                for lt in indices[r, :count].tolist():
                    pb = int(block_table[req, lt // page_size])
                    phys.append(slot_row[pb * page_size + lt % page_size])
                kk, vv = dk[phys].double(), dv[phys].double()
                for h in range(n_q):
                    kvh = h // (n_q // n_kv)
                    p = torch.softmax((q[r, h].double() @ kk[:, kvh].T) * head_dim**-0.5, dim=0)
                    o = (p @ vv[:, kvh]).to(torch.bfloat16).double()
                    ref[r, h] = o * torch.sigmoid(gate[r, h].double())
            err = float((got - ref).abs().max() / ref.abs().max())
            out_bf16 = torch.empty_like(q, device=device)
            ops.qsa_sparse_paged_attention(
                q.to(device), twin_k, twin_v, indices.to(device), block_table.to(device),
                token_to_req.to(device), prefill, out_bf16, output_gate=gate.to(device),
            )
            twin_ref = out_bf16.double().cpu()
            err_twin = float((got - twin_ref).abs().max() / twin_ref.abs().max())
            worst, worst_twin = max(worst, err), max(worst_twin, err_twin)
            # E062 on the CPU interpreter: 1.1e-2 / 6.6e-3 (bf16 output rounding, P in
            # fp16 vs bf16); a wrong layout or scale lookup gives errors near 1.
            ok_kernel = ok_kernel and math.isfinite(err) and err < 3e-2 and err_twin < 2e-2
            log(
                f"[e083-selftest] {layout_name} kernel rows={rows} prefill={prefill}: max err "
                f"vs float64 {err:.2e}, vs bf16 branch on the same values {err_twin:.2e}"
            )
        lv["kernel"] = bool(ok_kernel)
        log(f"[e083-selftest] {layout_name}: {json.dumps(lv)}")
        verdict["per_layout"][layout_name] = lv
        ok_all = ok_all and lv["writer"] and lv["isolate"] and lv["kernel"]
        for name in ("writer", "isolate", "kernel"):
            verdict["checks"][f"{name}_{layout_name}"] = lv[name]
    verdict["kernel_max_rel_err"] = worst
    verdict["kernel_vs_bf16_max_rel_err"] = worst_twin
    verdict["writer"] = "triton"
    verdict["ok"] = bool(ok_all and all(verdict["checks"].values()))
    return verdict


def selftest_main(out_path: str | None) -> int:
    """--selftest: run selftest() in the server's environment, before the server."""
    verdict: dict = {"ok": False}
    try:
        import torch

        try:  # importing the model first avoids a circular import of ops/qsa.py
            importlib.import_module("vllm.models.qwen4_exp.nvidia.model")
        except Exception:  # noqa: BLE001
            pass
        from vllm.models.qwen4_exp.nvidia.ops import qsa as ops
        from vllm.v1.kv_cache_interface import create_kv_cache_views
        from vllm.v1.kv_cache_layout import KVCacheLayout

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        if not hasattr(ops, "k8v4_cache_views"):
            raise RuntimeError("the E083 port is not applied to this runtime")
        native = None
        try:
            from vllm._custom_ops import reshape_and_cache_flash as native
        except Exception as exc:  # noqa: BLE001
            print(f"[e083-selftest] native fp8 writer unavailable ({exc!r}); keys checked against torch only")
        verdict = selftest(ops, create_kv_cache_views, KVCacheLayout, "cuda", native)
    except Exception as exc:  # noqa: BLE001
        verdict = {"ok": False, "error": repr(exc)[:1000]}
    verdict["recommend"] = (
        {"kv_cache_dtype": "fp8_e4m3", FLAG_ENV: "1"}
        if verdict.get("ok")
        else {"kv_cache_dtype": "fp8_e4m3", FLAG_ENV: "0"}
    )
    text = json.dumps(verdict, indent=1, default=str)
    print(text)
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    return 0 if verdict.get("ok") else 1


# ---------------------------------------------------------------------------
# Post-start check against the running server (--server-check). Same probes as
# E062's (so E061b's fp8 baseline file is directly comparable), plus an absolute
# NLL bar. Run it right after the K8V4 server is healthy; on a non-zero exit
# restart in plain fp8.
# ---------------------------------------------------------------------------


def _post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _filler(num_words: int, seed: int) -> list[str]:
    import random

    rng = random.Random(seed)
    nouns = ["grid", "cell", "agent", "level", "colour", "block", "wall", "door", "key", "move"]
    verbs = ["shifts", "opens", "blocks", "repeats", "mirrors", "fills", "turns", "hides"]
    words: list[str] = []
    while len(words) < num_words:
        words += f"The {rng.choice(nouns)} {rng.choice(verbs)} the {rng.choice(nouns)} near row {rng.randint(1, 64)}.".split()
    return words[:num_words]


def server_check(
    base_url: str,
    model: str,
    baseline: dict | None = None,
    nll_max: float | None = None,
    log=print,
) -> dict:
    """needle: a 6-digit code planted at 10/30/50/70/90 % depth of ~8k-60k token
    filler must come back (>= 4 of 5); nll: mean prompt NLL of a fixed English text
    (stdlib docstrings; E062's text, 2,935 tokens, fp8 baseline 0.03905 in E061b)
    must stay below nll_max when given (E083 kill bar 0.043), else below 1.03 x the
    fp8 baseline when given, else below 4 nats (garbage gives > 8); mtp: draft
    acceptance from /metrics must stay above 0.8 x baseline (else 0.25), which
    catches a broken cache in the MTP layer."""
    import argparse as _ap
    import json as _js
    import random

    verdict: dict = {"ok": False}
    hits = 0
    for i, (words, depth) in enumerate(((6000, 0.1), (20000, 0.3), (45000, 0.5), (20000, 0.7), (6000, 0.9))):
        code = str(random.Random(100 + i).randint(100000, 999999))
        filler = _filler(words, i)
        cut = int(len(filler) * depth)
        text = " ".join(filler[:cut] + [f"Remember this: the secret code is {code}."] + filler[cut:])
        resp = _post(f"{base_url}/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": text + "\n\nWhat is the secret code? Answer with the number only."}],
            "temperature": 0.0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        msg = resp["choices"][0]["message"]
        answer = (msg.get("content") or "") + (msg.get("reasoning_content") or msg.get("reasoning") or "")
        hits += code in answer
        log(f"[e083-check] needle {i}: {words} words at {depth:.0%}: {'hit' if code in answer else 'MISS'}")
    verdict["needle_hits"] = hits

    # Same text as E062's check (its docstring sources must stay these modules).
    text = "\n\n".join(m.__doc__ or "" for m in (_ap, _js, json, os, random))
    resp = _post(f"{base_url}/v1/completions", {
        "model": model, "prompt": text, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0,
    })
    lps = []
    for entry in resp["choices"][0].get("prompt_logprobs") or []:
        if entry:
            best = next(iter(entry.values()))
            lp = best.get("logprob") if isinstance(best, dict) else best
            if lp is not None:
                lps.append(-float(lp))
    nll = sum(lps) / max(1, len(lps))
    verdict["prompt_tokens"], verdict["mean_nll"] = len(lps), nll
    if nll_max is not None:
        nll_bound = float(nll_max)
    elif baseline and baseline.get("mean_nll"):
        nll_bound = 1.03 * baseline["mean_nll"]
    else:
        nll_bound = 4.0
    verdict["nll_bound"] = nll_bound
    if baseline and baseline.get("mean_nll"):
        verdict["nll_vs_baseline"] = nll / baseline["mean_nll"]

    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as r:
            metrics = r.read().decode("utf-8")
        acc = drafts = 0.0
        for line in metrics.splitlines():
            if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                acc += float(line.rsplit(" ", 1)[1])
            elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
                drafts += float(line.rsplit(" ", 1)[1])
        verdict["mtp_acceptance"] = acc / drafts if drafts else None
    except Exception as exc:  # noqa: BLE001
        verdict["mtp_acceptance"] = None
        verdict["metrics_error"] = repr(exc)[:200]
    mtp_floor = 0.8 * baseline["mtp_acceptance"] if baseline and baseline.get("mtp_acceptance") else 0.25
    verdict["checks"] = {
        "needle": hits >= 4,
        "nll": len(lps) > 100 and nll <= nll_bound,
        "mtp": verdict["mtp_acceptance"] is None or verdict["mtp_acceptance"] >= mtp_floor,
    }
    verdict["ok"] = all(verdict["checks"].values())
    log(f"[e083-check] needle {hits}/5, mean NLL {nll:.4f} (bound {nll_bound:.4f}), "
        f"MTP acceptance {verdict['mtp_acceptance']} (floor {mtp_floor:.2f}) -> ok={verdict['ok']}")
    return verdict


def server_check_main(
    base_url: str, model: str, baseline_path: str | None, nll_max: float | None, out_path: str | None
) -> int:
    baseline = None
    if baseline_path and Path(baseline_path).is_file():
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    try:
        verdict = server_check(base_url.rstrip("/"), model, baseline, nll_max)
    except Exception as exc:  # noqa: BLE001
        verdict = {"ok": False, "error": repr(exc)[:1000]}
    text = json.dumps(verdict, indent=1, default=str)
    print(text)
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    return 0 if verdict.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--site", help="directory that contains the vllm package (or ARC3_KVQ_SITE)")
    p.add_argument("--overlay-identity", help="PATCH_IDENTITY.json of the runtime dataset")
    p.add_argument("--check", action="store_true", help="verify pins and anchors only")
    p.add_argument("--selftest", action="store_true", help="GPU kernel self-test (server env)")
    p.add_argument("--out", help="verdict JSON path (self-test or server check)")
    p.add_argument("--server-check", metavar="BASE_URL", help="post-start check of a running server")
    p.add_argument("--model", help="served model name (server check)")
    p.add_argument("--baseline", help="server-check verdict JSON of the fp8 server (optional)")
    p.add_argument("--nll-max", type=float, help="absolute prompt-NLL bar (E083 kill bar: 0.043)")
    args = p.parse_args(argv)
    if args.selftest:
        return selftest_main(args.out)
    if args.server_check:
        return server_check_main(args.server_check, args.model or "", args.baseline, args.nll_max, args.out)
    return run(args.site, args.overlay_identity, args.check)


if __name__ == "__main__":
    sys.exit(main())
