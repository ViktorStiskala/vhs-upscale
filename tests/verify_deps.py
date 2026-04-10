#!/usr/bin/env python3
"""
Verify VHS restoration pipeline dependencies inside Docker.

Runs against the actual installed packages (not source repos) to check:
1. Python imports succeed
2. Function/class signatures have required kwargs (inspect.signature)
3. VapourSynth native plugins load with expected namespaces
4. Filesystem paths exist (models, plugins, scripts, binaries)
5. Path defaults are consistent across upscale.vpy, upscale.sh, Dockerfile

Usage (inside Docker):
    python3 verify_deps.py /path/to/source/files

Build & run:
    docker build -t vhs-restore .
    docker build -f tests/Dockerfile --build-arg BASE_IMAGE=vhs-restore -t vhs-test .
"""

import importlib
import inspect
import os
import re
import sys

# ============================================================
# Results collector
# ============================================================
passes = []
warnings = []
errors = []


def ok(msg):
    passes.append(msg)
    print(f"  PASS  {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  WARN  {msg}")


def fail(msg):
    errors.append(msg)
    print(f"  FAIL  {msg}")


# ============================================================
# 1. Import checks
# ============================================================
REQUIRED_IMPORTS = [
    ("vapoursynth", "VapourSynth core"),
    ("torch", "PyTorch"),
    ("vsspandrel", "AI model inference (vsspandrel)"),
    ("vsdeinterlace", "QTGMC deinterlacer (vsjetpack)"),
    ("vsdenoise", "DFTTest denoiser (vsjetpack)"),
    ("vsscunet", "SCUNet denoiser"),
    ("spandrel", "Model format support"),
]


def check_imports():
    print("=== Import checks ===\n")
    for module_name, description in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            ok(f"import {module_name} ({description})")
        except ImportError as e:
            fail(f"import {module_name} ({description}): {e}")


# ============================================================
# 2. API signature checks (inspect real installed code)
# ============================================================
SIGNATURE_CHECKS = [
    {
        "import": "from vsspandrel import vsspandrel",
        "callable": "vsspandrel.vsspandrel",
        "required_params": ["model_path", "trt"],
        "description": "vsspandrel(clip, model_path=..., trt=False) — called 4x in pipeline",
    },
    {
        "import": "from vsdeinterlace import QTempGaussMC",
        "callable": "vsdeinterlace.QTempGaussMC",
        "required_params": [
            "sharpen_strength",
            "denoise_func",
            "denoise_mc_denoise",
            "basic_noise_restore",
        ],
        "description": "QTempGaussMC.__init__ kwargs (upscale.vpy:148-152)",
    },
    {
        "import": "from vsdeinterlace import QTempGaussMC",
        "callable": "vsdeinterlace.QTempGaussMC.bob",
        "required_params": ["tff"],
        "description": "QTempGaussMC.bob(clip, tff=...) (upscale.vpy:153)",
    },
    {
        "import": "from vsdenoise import DFTTest",
        "callable": "vsdenoise.DFTTest",
        "required_params": ["sigma"],
        "description": "DFTTest(sigma=3.0) temporal denoise (upscale.vpy:150)",
    },
    {
        "import": "from vsscunet import scunet",
        "callable": "vsscunet.scunet",
        "required_params": ["model", "trt"],
        "description": "scunet(clip, model=..., trt=False) (upscale.vpy:211)",
    },
]

ENUM_CHECKS = [
    {
        "import": "from vsscunet import SCUNetModel",
        "class": "vsscunet.SCUNetModel",
        "required_members": ["scunet_color_real_psnr"],
        "description": "SCUNetModel.scunet_color_real_psnr enum (upscale.vpy:211)",
    },
]


def resolve_callable(dotted_path):
    """Resolve 'module.Class.method' to the actual callable."""
    parts = dotted_path.split(".")
    obj = importlib.import_module(parts[0])
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj


def get_all_params(callable_obj):
    """Get all parameter names from a callable, handling **kwargs."""
    try:
        sig = inspect.signature(callable_obj)
    except (ValueError, TypeError):
        return None, False

    params = set()
    has_var_keyword = False
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
        else:
            params.add(name)
    return params, has_var_keyword


def check_signatures():
    print("\n=== API signature checks ===\n")

    for check in SIGNATURE_CHECKS:
        desc = check["description"]
        try:
            obj = resolve_callable(check["callable"])
        except (ImportError, AttributeError) as e:
            fail(f"{desc}: {e}")
            continue

        params, has_var_keyword = get_all_params(obj)
        if params is None:
            warn(f"{desc}: could not inspect signature")
            continue

        required = set(check["required_params"])
        missing = required - params
        if not missing:
            ok(desc)
        elif has_var_keyword:
            warn(f"{desc}: has **kwargs, params {missing} may be passed through")
        else:
            fail(f"{desc}: missing params {missing}. Has: {sorted(params)}")

    for check in ENUM_CHECKS:
        desc = check["description"]
        try:
            cls = resolve_callable(check["class"])
        except (ImportError, AttributeError) as e:
            fail(f"{desc}: {e}")
            continue

        missing = []
        for member in check["required_members"]:
            if not hasattr(cls, member):
                missing.append(member)
        if not missing:
            ok(desc)
        else:
            fail(f"{desc}: missing members {missing}")


# ============================================================
# 3. VapourSynth plugin checks
# ============================================================
REQUIRED_PLUGINS = ["mv", "fmtc", "znedi3", "dfttest", "resize2", "akarin", "noise"]
OPTIONAL_PLUGINS = [
    ("nlm_cuda", "CUDA NLM denoiser (preferred)"),
    ("knlm", "OpenCL NLM denoiser (fallback)"),
    ("cas", "CAS sharpening"),
]
SOURCE_PLUGINS = ["ffms2", "bs", "lsmas"]


def check_vs_plugins():
    print("\n=== VapourSynth plugin checks ===\n")
    try:
        import vapoursynth as vs

        core = vs.core
    except ImportError:
        fail("VapourSynth not importable — cannot check plugins")
        return

    loaded = sorted([p for p in dir(core) if not p.startswith("_")])

    for plugin in REQUIRED_PLUGINS:
        if plugin in loaded:
            ok(f"core.{plugin}")
        else:
            fail(f"core.{plugin} — required plugin not loaded")

    for plugin, desc in OPTIONAL_PLUGINS:
        if plugin in loaded:
            ok(f"core.{plugin} ({desc})")
        else:
            warn(f"core.{plugin} not loaded ({desc})")

    source_found = [p for p in SOURCE_PLUGINS if p in loaded]
    if source_found:
        ok(f"Source plugin(s): {', '.join(source_found)}")
    else:
        fail("No source plugin loaded (need ffms2, bs, or lsmas)")

    # Check NLM fallback — at least one must exist
    has_nlm = "nlm_cuda" in loaded or "knlm" in loaded
    if not has_nlm:
        warn("No NLM denoiser available — chroma denoise will be skipped")


# ============================================================
# 4. Filesystem path checks
# ============================================================
REQUIRED_PATHS = [
    ("/models/scunet_color_real_psnr.pth", "SCUNet model"),
    ("/models/1x-BleedOut-Compact.pth", "BleedOut model"),
    ("/models/2xLiveActionV1_SPAN.pth", "SPAN model"),
    ("/usr/local/lib/vapoursynth", "VS plugin directory"),
    ("/opt/vhs-restore/upscale.vpy", "Pipeline script"),
    ("/opt/vhs-restore/upscale.sh", "Entry point script"),
    ("/opt/vhs-restore/batch_upscale.sh", "Batch script"),
    ("/opt/vhs-restore/setup.sh", "Setup script"),
]

REQUIRED_BINARIES = ["ffmpeg", "ffprobe", "python3"]
OPTIONAL_BINARIES = ["vspipe"]


def check_paths():
    print("\n=== Filesystem path checks ===\n")

    for path, desc in REQUIRED_PATHS:
        if os.path.exists(path):
            ok(f"{path} ({desc})")
        else:
            fail(f"{path} missing ({desc})")

    # Check VS plugin dir has .so files
    vs_plugin_dir = "/usr/local/lib/vapoursynth"
    if os.path.isdir(vs_plugin_dir):
        so_files = [f for f in os.listdir(vs_plugin_dir) if f.endswith(".so")]
        if so_files:
            ok(f"{vs_plugin_dir}/ has {len(so_files)} .so files")
        else:
            fail(f"{vs_plugin_dir}/ exists but contains no .so files")

    for binary in REQUIRED_BINARIES:
        from shutil import which

        if which(binary):
            ok(f"{binary} in PATH")
        else:
            fail(f"{binary} not found in PATH")

    for binary in OPTIONAL_BINARIES:
        from shutil import which

        if which(binary):
            ok(f"{binary} in PATH")
        else:
            warn(f"{binary} not in PATH (Python fallback will be used)")

    # Check VAPOURSYNTH_EXTRA_PLUGIN_PATH env
    vs_env = os.environ.get("VAPOURSYNTH_EXTRA_PLUGIN_PATH", "")
    if vs_env:
        if os.path.isdir(vs_env):
            ok(f"VAPOURSYNTH_EXTRA_PLUGIN_PATH={vs_env} (exists)")
        else:
            fail(f"VAPOURSYNTH_EXTRA_PLUGIN_PATH={vs_env} (directory missing)")
    else:
        warn("VAPOURSYNTH_EXTRA_PLUGIN_PATH not set")


# ============================================================
# 5. Source file path consistency
# ============================================================
PATH_CONSISTENCY = [
    {
        "name": "MODEL_DIR",
        "vpy_pattern": r'os\.environ\.get\(\s*"MODEL_DIR"\s*,\s*"([^"]+)"\s*\)',
        "sh_pattern": r'MODEL_DIR="\$\{MODEL_DIR:-([^}]+)\}"',
    },
    {
        "name": "USER_MODEL_DIR",
        "vpy_pattern": r'os\.environ\.get\(\s*"USER_MODEL_DIR"\s*,\s*"([^"]+)"\s*\)',
        "sh_pattern": r'USER_MODEL_DIR="\$\{USER_MODEL_DIR:-([^}]+)\}"',
    },
]


def check_source_consistency(src_dir):
    print("\n=== Source file path consistency ===\n")

    vpy_path = os.path.join(src_dir, "upscale.vpy")
    sh_path = os.path.join(src_dir, "upscale.sh")
    dockerfile_path = os.path.join(src_dir, "Dockerfile")

    missing = []
    for path in [vpy_path, sh_path, dockerfile_path]:
        if not os.path.exists(path):
            missing.append(os.path.basename(path))
    if missing:
        warn(f"Source files not available for consistency check: {missing}")
        return

    vpy_text = open(vpy_path).read()
    sh_text = open(sh_path).read()
    dockerfile_text = open(dockerfile_path).read()

    for check in PATH_CONSISTENCY:
        name = check["name"]
        vpy_match = re.search(check["vpy_pattern"], vpy_text)
        sh_match = re.search(check["sh_pattern"], sh_text)

        vpy_val = vpy_match.group(1) if vpy_match else None
        sh_val = sh_match.group(1) if sh_match else None

        if vpy_val is None:
            warn(f"{name}: default not found in upscale.vpy")
            continue
        if sh_val is None:
            warn(f"{name}: default not found in upscale.sh")
            continue

        if vpy_val != sh_val:
            fail(f"{name}: mismatch — upscale.vpy='{vpy_val}' vs upscale.sh='{sh_val}'")
        else:
            ok(f"{name}: consistent ('{vpy_val}')")

    # Plugin path: COPY dest must match ENV
    plugin_copy = re.search(
        r"COPY\s+--from=stage-plugins\s+\S+\s+(/\S+)", dockerfile_text
    )
    plugin_env = re.search(
        r"ENV\s+VAPOURSYNTH_EXTRA_PLUGIN_PATH=(\S+)", dockerfile_text
    )
    if plugin_copy and plugin_env:
        copy_dest = plugin_copy.group(1).rstrip("/")
        env_val = plugin_env.group(1).rstrip("/")
        if copy_dest == env_val:
            ok(f"Plugin path: consistent ('{env_val}')")
        else:
            fail(f"Plugin path: COPY dest='{copy_dest}' vs ENV='{env_val}'")

    # Script path: COPY dest dir must match PATH env
    script_copies = re.findall(r"COPY\s+\S+\.(?:vpy|sh)\s+(/\S+)", dockerfile_text)
    path_env = re.search(r'ENV\s+PATH="([^:]+):', dockerfile_text)
    if script_copies and path_env:
        script_dir = os.path.dirname(script_copies[0])
        env_path = path_env.group(1)
        if script_dir == env_path:
            ok(f"Script path: consistent ('{env_path}')")
        else:
            fail(f"Script path: COPY dir='{script_dir}' vs PATH='{env_path}'")


# ============================================================
# Main
# ============================================================
def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else None

    check_imports()
    check_signatures()
    check_vs_plugins()
    check_paths()

    if src_dir:
        check_source_consistency(src_dir)
    else:
        print("\n=== Source consistency: skipped (no src dir provided) ===")

    # Summary
    total = len(passes) + len(warnings) + len(errors)
    print(
        f"\n=== Summary: {len(passes)} passed, {len(warnings)} warnings, "
        f"{len(errors)} errors (of {total} checks) ==="
    )

    if errors:
        print("\nERRORS (will break at runtime):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    elif warnings:
        print("\nAll critical checks passed. Review warnings above.")
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
