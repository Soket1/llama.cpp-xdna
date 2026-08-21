"""CPU-only tests for FlowKV's compile-time specialization contract.

These tests import only ``flowkv_decode.contract`` and inspect wrapper source text;
they neither import AIE runtime modules nor resolve an NPU device.
"""

from pathlib import Path
import sys

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[4]
_IRON_ROOT = _REPO_ROOT / "IRON-windows"
sys.path.insert(0, str(_IRON_ROOT))

from iron.operators.flowkv_decode.contract import (  # noqa: E402
    FlowKVContractError,
    flowkv_artifact_identity,
    flowkv_bundle_cache_key,
    flowkv_effective_chunk,
    flowkv_geometry_flags,
    flowkv_is_npu1_device,
    flowkv_kernel_object_name,
    flowkv_tuning_fingerprint,
    flowkv_tuning_tokens_from_environment,
    normalize_flowkv_tuning_tokens,
)


class _NamedDevice:
    def __init__(self, name):
        self.name = name


class _ResolvableDevice:
    def __init__(self, name):
        self._resolved = _NamedDevice(name)

    def resolve(self):
        return self._resolved


@pytest.mark.parametrize(
    ("seq_len", "expected"),
    [(32, 32), (64, 32), (256, 32), (33, 33), (257, 257)],
)
def test_effective_chunk_matches_live_wrapper_policy(seq_len, expected):
    assert flowkv_effective_chunk(seq_len) == expected


@pytest.mark.parametrize("value", [0, -1, True, "32"])
def test_effective_chunk_rejects_non_positive_integer_geometry(value):
    with pytest.raises(FlowKVContractError):
        flowkv_effective_chunk(value)


def test_geometry_flags_and_default_object_identity_cover_3b_geometry():
    assert flowkv_geometry_flags(128, 3, 256) == (
        "-DHEAD_DIM=128",
        "-DMAX_Q_HEADS=3",
        "-DMAX_CHUNK=256",
    )
    assert flowkv_kernel_object_name(128, 3, 256) == "flowkv_128d_h3_c256.o"


@pytest.mark.parametrize("value", [0, -1, True, "128"])
def test_geometry_flags_reject_invalid_capacity(value):
    with pytest.raises(FlowKVContractError):
        flowkv_geometry_flags(value, 3, 256)


def test_tuning_normalization_is_order_independent_and_cache_safe():
    first = ("-DFLOWKV_VEC_EXP=1", "-DFLOWKV_NOEXP=1")
    second = ("-D", "FLOWKV_NOEXP=1", "-DFLOWKV_VEC_EXP=1")
    expected = ("-DFLOWKV_NOEXP=1", "-DFLOWKV_VEC_EXP=1")

    assert normalize_flowkv_tuning_tokens(first) == expected
    assert normalize_flowkv_tuning_tokens(second) == expected
    assert flowkv_tuning_fingerprint(first) == flowkv_tuning_fingerprint(second)
    assert flowkv_kernel_object_name(128, 3, 256, first) == (
        "flowkv_128d_h3_c256_t"
        f"{flowkv_tuning_fingerprint(expected)}.o"
    )


@pytest.mark.parametrize(
    "tokens",
    [
        ("-DHEAD_DIM=128",),
        ("-D", "MAX_Q_HEADS=3"),
        ("-DMAX_CHUNK256",),
        ("-UHEAD_DIM",),
        ("-U", "MAX_CHUNK"),
        ("-DFLOWKV_NOEXP=0",),
        ("-DFLOWKV_NOEXP=1", "-DFLOWKV_NOEXP=1"),
        ("-DUNRELATED_MACRO=1",),
        ("-DFLOWKV_NOEXP=1;unexpected",),
    ],
)
def test_tuning_rejects_geometry_overrides_and_unsafe_definitions(tokens):
    with pytest.raises(FlowKVContractError):
        normalize_flowkv_tuning_tokens(tokens)


def test_empty_tuning_identity_is_pinned_and_default_object_has_no_suffix():
    assert flowkv_tuning_fingerprint(()) == "63624e81faf2e175"
    assert flowkv_artifact_identity(64, 4, 32) == "63624e81faf2e175"
    assert flowkv_kernel_object_name(64, 4, 32) == "flowkv_64d_h4_c32.o"


def test_bundle_cache_key_is_native_compatible_and_tuning_aware():
    default = flowkv_bundle_cache_key(24, 8, 128, 256, 256, 8)
    tuned = flowkv_bundle_cache_key(
        24, 8, 128, 256, 256, 8, "fe2d60c9196a71df"
    )

    assert default == "flowkv_H24_KV8_d128_S256_C256_8col_t63624e81faf2e175"
    assert tuned == "flowkv_H24_KV8_d128_S256_C256_8col_tfe2d60c9196a71df"
    assert len({
        default,
        flowkv_bundle_cache_key(24, 8, 64, 256, 256, 8),
        flowkv_bundle_cache_key(24, 8, 128, 128, 256, 8),
        flowkv_bundle_cache_key(24, 8, 128, 256, 32, 8),
        flowkv_bundle_cache_key(24, 8, 128, 256, 256, 4),
        tuned,
    }) == 6


@pytest.mark.parametrize("fingerprint", ["", "ABCDEF0123456789", "0123456789abcdeg", 1])
def test_bundle_cache_key_rejects_invalid_fingerprint(fingerprint):
    with pytest.raises(FlowKVContractError):
        flowkv_bundle_cache_key(24, 8, 128, 256, 256, 8, fingerprint)


def test_environment_tuning_is_normalized_before_identity(monkeypatch):
    monkeypatch.setenv("FLOWKV_CFLAGS", "-DFLOWKV_VEC_EXP=1 -DFLOWKV_NOEXP=1")
    assert flowkv_tuning_tokens_from_environment() == (
        "-DFLOWKV_NOEXP=1",
        "-DFLOWKV_VEC_EXP=1",
    )


def test_npu1_device_detection_supports_strings_and_resolved_device_objects():
    assert flowkv_is_npu1_device("npu")
    assert flowkv_is_npu1_device("npu1")
    assert flowkv_is_npu1_device(_ResolvableDevice("npu"))
    assert flowkv_is_npu1_device(_ResolvableDevice("npu1"))
    assert not flowkv_is_npu1_device("npu2")
    assert not flowkv_is_npu1_device(_ResolvableDevice("npu2"))


def test_native_flowkv_compile_path_uses_argv_and_serializes_loads():
    source = (_REPO_ROOT / "ggml/src/ggml-xdna/ggml-xdna.cpp").read_text(
        encoding="utf-8"
    )

    assert "#include <filesystem>" in source
    assert "#include <sheredom/subprocess.h>" in source
    assert '"flowkv_H%lld_KV%lld_d%lld_S%lld_C%lld_%dcol_t%s"' in source
    assert "static std::string make_flowkv_cache_key(" in source
    assert "static constexpr const char * FLOWKV_DEFAULT_TUNING_FINGERPRINT" in source
    assert "static bool xdna_split_command(" in source
    assert "static int xdna_run_process(" in source
    assert "subprocess_option_combined_stdout_stderr" in source
    assert "process.stdin_file = nullptr;" in source
    assert "xdna_print_process_output(subprocess_stdout(&process));" in source

    ensure_start = source.index("static bool ensure_flowkv_compiled(")
    ensure_end = source.index("static bool try_load_flowkv_entry(", ensure_start)
    ensure_source = source[ensure_start:ensure_end]
    assert "std::vector<std::string> compiler_argv;" in ensure_source
    assert "xdna_split_command(xdna_python_cmd(), &compiler_argv)" in ensure_source
    assert '"--flowkv-cflag=" + token' in ensure_source
    assert 'compiler_argv.insert(compiler_argv.end(), {"--out", bundle_dir});' in ensure_source
    assert "int ret = xdna_run_process(compiler_argv);" in ensure_source
    assert "system(" not in ensure_source
    assert ensure_source.count("ctx->flowkv_compile_failed") == 4
    assert ensure_source.count("std::lock_guard<std::mutex> lock(ctx->cache_mutex);") == 4

    load_start = source.index("static xdna_flowkv_entry * get_or_load_flowkv_kernel(")
    load_end = source.index("// ============================================================================\n// Expanded decode-attention", load_start)
    load_source = source[load_start:load_end]
    assert "static std::mutex flowkv_compile_load_mutex;" in load_source
    assert "std::unique_lock<std::mutex> compile_load_lock(flowkv_compile_load_mutex);" in load_source
    assert "std::filesystem::remove_all(bundle_dir, cleanup_ec);" in load_source
    assert 'std::string rm_cmd = "rmdir /s /q' not in load_source
    assert 'std::string rm_cmd = "rm -rf' not in load_source
    assert "system(" not in load_source
    assert load_source.count("ctx->flowkv_compile_failed") == 4
    assert load_source.count("std::lock_guard<std::mutex> lock(ctx->cache_mutex);") == 5


def test_flowkv_kernel_source_requires_all_static_geometry_macros():
    source = (_IRON_ROOT / "aie_kernels/aie2p/flowkv.cc").read_text(encoding="utf-8")

    for macro in ("HEAD_DIM", "MAX_Q_HEADS", "MAX_CHUNK"):
        assert f"#ifndef {macro}" in source
        assert f'#error "FlowKV: {macro} must be passed via -D{macro}=N' in source

    assert "#define HEAD_DIM 64" not in source


def test_front_attn_preserves_its_r4_topology_suffix_on_shared_identity():
    op_source = (_IRON_ROOT / "iron/operators/decode_front_attn/op.py").read_text(
        encoding="utf-8"
    )
    design_source = (
        _IRON_ROOT / "iron/operators/decode_front_attn/design.py"
    ).read_text(encoding="utf-8")

    for source in (op_source, design_source):
        assert "flowkv_effective_chunk" in source
        assert "flowkv_kernel_object_name" in source
        assert '.replace(".o", "_r4.o")' in source
    assert "flowkv_flags = flowkv_geometry_flags(" in op_source
    assert "extra_flags=list(flowkv_flags)" in op_source


def test_f3best_passes_tuned_flowkv_identity_through_mlir_emitter():
    op_source = (_IRON_ROOT / "iron/operators/decode_layer_f3best/op.py").read_text(
        encoding="utf-8"
    )
    design_source = (
        _IRON_ROOT / "iron/operators/decode_layer_f3best/design.py"
    ).read_text(encoding="utf-8")
    no_kv_emitter_source = (
        _IRON_ROOT / "iron/operators/decode_layer_f3best/f3best_emit_nokv.py"
    ).read_text(encoding="utf-8")
    kv_emitter_source = (
        _IRON_ROOT / "iron/operators/decode_layer_f3best/f3best_emit.py"
    ).read_text(encoding="utf-8")

    assert "flowkv_tuning = (" in op_source
    assert '"-DFLOWKV_PRESCALE_Q=1"' in op_source
    assert '"-DFLOWKV_VEC_EXP=1"' in op_source
    assert '"-DFLOWKV_VALUE_AMAC=1"' in op_source
    assert "flowkv_obj_name = flowkv_kernel_object_name(" in op_source
    assert "flowkv_obj_tag = f\"_fkobj_{flowkv_obj_name.removesuffix('.o')}\"" in op_source
    assert "_uni_sfx}{flowkv_obj_tag}_fkfix2" in op_source
    assert op_source.index("flowkv_obj_name = flowkv_kernel_object_name(") < op_source.index(
        "base = ("
    )
    assert "flowkv_flags = flowkv_geometry_flags(" in op_source
    assert "flowkv_obj_name," in op_source
    assert "extra_flags=[*flowkv_flags, *flowkv_tuning]" in op_source
    assert 'rope_obj_name = f"rope_il_k{self.K}_d{self.head_dim}.o"' in op_source
    assert 'f"layer_fused_relay_e{E}_h{H}_g{g}_d{self.head_dim}"' in op_source
    assert 'f"_qh{self.num_q_heads}_kvh{self.num_kv_heads}_s2048_c8_m1024.o"' in op_source
    assert "rope_obj = KernelObjectArtifact.new(\n            rope_obj_name," in op_source
    assert "relay_obj = KernelObjectArtifact.new(\n            relay_obj_name," in op_source
    assert "rope_obj_name, relay_obj_name," in op_source
    assert "_objid2_abi3_al64" in op_source

    native_source = (_REPO_ROOT / "ggml/src/ggml-xdna/ggml-xdna.cpp").read_text(
        encoding="utf-8"
    )
    f3best_start = native_source.index("} else if (op_kind == XDNA_OP_DECODE_LAYER_F3BEST) {")
    f3best_end = native_source.index("    } else {", f3best_start)
    f3best_source = native_source[f3best_start:f3best_end]
    # #267i: the native mirror now COMPUTES the fingerprint with the same
    # canonical form and FNV-1a constants as flowkv_tuning_fingerprint instead of
    # hardcoding hex strings, so adding a tuning macro can no longer desynchronize
    # the two cache keys (which is how #266's first A/B came back bit-identical).
    assert '"flowkv-cflags-v1;"' in f3best_source
    assert "0xCBF29CE484222325ull" in f3best_source
    assert "0x100000001B3ull" in f3best_source
    assert "std::sort(fk_tokens.begin(), fk_tokens.end())" in f3best_source
    for token in (
        "-DFLOWKV_PRESCALE_Q=1",
        "-DFLOWKV_VEC_EXP=1",
        "-DFLOWKV_VALUE_AMAC=1",
        "-DFLOWKV_VALUE_LEGACY=1",
        "-DFLOWKV_DOT_MULINIT=1",
    ):
        assert f'"{token}"' in f3best_source
        assert f'"{token}"' in op_source or token in op_source
    assert '"_fkobj_flowkv_%lldd_h%lld_c256_t%s"' in f3best_source
    assert "%s_fkfix2_silu2_mxpp_objid2_abi3_al64" in f3best_source
    # #267l/n: the 16-wide block exp2 loop is the root of #187 and is correct
    # only at head_dim 64, so the production arm is selected BY GEOMETRY in both
    # op.py and the native cache key. If the two selections drift apart the
    # probe silently links a different FlowKV object than the xclbin name claims.
    assert "elif self.head_dim == 64:" in op_source
    assert "else if (head_dim == 64)" in f3best_source
    assert "FLOWKV_VEC_EXP_DIRECT" not in op_source
    assert "FLOWKV_VEC_EXP_DIRECT" not in f3best_source

    assert "with_npu_kv=False, flowkv_obj_name=None, rope_obj_name=None," in design_source
    assert "relay_obj_name=None):" in design_source
    assert "flowkv_obj_name=flowkv_obj_name" in design_source
    assert "rope_obj_name=rope_obj_name, relay_obj_name=relay_obj_name" in design_source
    assert "POS=5, flowkv_obj_name=None, rope_obj_name=None, relay_obj_name=None" in no_kv_emitter_source
    assert "self.ROPE_LIB = rope_obj_name if rope_obj_name is not None else \"rope_il.o\"" in no_kv_emitter_source
    assert 'link_with = "{self.ROPE_LIB}"' in no_kv_emitter_source
    assert 'link_with = "{self.RELAY_LIB}"' in no_kv_emitter_source
    assert 'link_with = "rope_il.o"' not in no_kv_emitter_source
    assert 'link_with = "layer_fused_relay.o"' not in no_kv_emitter_source
    assert "POS=5, flowkv_obj_name=None, rope_obj_name=None, relay_obj_name=None" in kv_emitter_source
    assert "flowkv_obj_name=flowkv_obj_name, rope_obj_name=rope_obj_name" in kv_emitter_source
    assert "relay_obj_name=relay_obj_name" in kv_emitter_source
    assert 'link_with = "{self.ROPE_LIB}"' in kv_emitter_source
    assert 'link_with = "{self.RELAY_LIB}"' in kv_emitter_source
    assert 'link_with = "rope_il.o"' not in kv_emitter_source
    assert 'link_with = "layer_fused_relay.o"' not in kv_emitter_source


def test_f3best_uses_short_temporary_build_root_and_stages_public_cache_pair():
    compile_source = (_REPO_ROOT / "ggml/src/ggml-xdna/compile.py").read_text(
        encoding="utf-8"
    )
    start = compile_source.index("def compile_decode_layer_f3best(")
    end = compile_source.index("\ndef compile_swiglu_decode(", start)
    f3best_source = compile_source[start:end]

    assert "import tempfile" in compile_source
    assert "tempfile.TemporaryDirectory(prefix=\"f3b_\", dir=build_parent)" in f3best_source
    assert "output_abs = os.path.abspath(output_path)" in f3best_source
    assert "output_drive, _ = os.path.splitdrive(output_abs)" in f3best_source
    assert 'build_parent = output_drive + os.path.sep if output_drive else tempfile.gettempdir()' in f3best_source
    assert "layer_f3best_build" not in f3best_source
    assert "context=AIEContext(build_dir=build_root)" in f3best_source
    assert f3best_source.index("op.compile()") < f3best_source.index(
        "shutil.copy2(str(compiled_xclbin), output_path)"
    )
    assert "shutil.copy2(str(compiled_xclbin), output_path)" in f3best_source
    assert 'insts_output = output_path.replace(".xclbin", ".insts")' in f3best_source
    assert "shutil.copy2(str(compiled_insts), insts_output)" in f3best_source
    assert f3best_source.index("with tempfile.TemporaryDirectory") < f3best_source.index(
        "return output_path"
    )


def test_standalone_flowkv_propagates_its_exact_specialization_to_design():
    op_source = (_IRON_ROOT / "iron/operators/flowkv_decode/op.py").read_text(
        encoding="utf-8"
    )
    design_source = (
        _IRON_ROOT / "iron/operators/flowkv_decode/design.py"
    ).read_text(encoding="utf-8")

    assert "self.kernel_obj_name = flowkv_kernel_object_name(" in op_source
    assert "tuning_fingerprint = flowkv_artifact_identity(" in op_source
    assert "self.kernel_obj_name," in op_source
    assert "KernelObjectArtifact.new(\n                    self.kernel_obj_name," in op_source
    assert "*flowkv_geometry_flags(" in op_source
    assert "*self.tuning_tokens," in op_source

    assert "kernel_obj_name=None," in design_source
    assert (
        'kernel_obj_name = f"flowkv_{head_dim}d_h{group_size}_c{chunk_size}.o"'
        in design_source
    )
    assert "kernel_obj = kernel_obj_name" in design_source
    assert design_source.count("kernel_obj,") == 6


@pytest.mark.parametrize(
    "relative_path",
    [
        "iron/operators/decode_front/op.py",
        "iron/operators/decode_layer/op.py",
        "iron/operators/decode_attn_split/op.py",
        "iron/operators/decode_attn_spatial/op.py",
        "iron/operators/decode_attn_oproj/op.py",
    ],
)
def test_active_wrapper_ops_share_canonical_geometry_contract(relative_path):
    source = (_IRON_ROOT / relative_path).read_text(encoding="utf-8")
    assert "flowkv_geometry_flags" in source
    assert "flowkv_kernel_object_name" in source
    assert "flowkv_obj_name = flowkv_kernel_object_name(" in source
    assert "flowkv_flags = flowkv_geometry_flags(" in source
    assert "extra_flags=list(flowkv_flags)" in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "iron/operators/decode_attn_split/op.py",
        "iron/operators/decode_attn_spatial/op.py",
        "iron/operators/decode_attn_oproj/op.py",
    ],
)
def test_chunked_wrapper_ops_use_shared_effective_chunk_policy(relative_path):
    source = (_IRON_ROOT / relative_path).read_text(encoding="utf-8")
    assert "flowkv_effective_chunk" in source
    assert "chunk_size = flowkv_effective_chunk(self.seq_len)" in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "iron/operators/flowkv_decode/design.py",
        "iron/operators/decode_front/design.py",
        "iron/operators/decode_layer/design.py",
        "iron/operators/decode_attn_split/design.py",
        "iron/operators/decode_attn_spatial/design.py",
        "iron/operators/decode_attn_oproj/design.py",
    ],
)
def test_flowkv_designs_normalize_device_objects_before_selecting_npu(relative_path):
    source = (_IRON_ROOT / relative_path).read_text(encoding="utf-8")
    assert "flowkv_is_npu1_device" in source
    assert "NPU1() if flowkv_is_npu1_device(dev) else NPU2()" in source
