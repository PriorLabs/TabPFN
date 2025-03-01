# ruff: noqa
"""This file is taken from PyTorch and modified to work with TabPFN, also
inspired from sklearn's show_versions function. This collects useful debug
information that can be used to report issues.

Usage: import tabpfn; tabpfn.show_versions().
"""

from __future__ import annotations

# mypy: allow-untyped-defs
import datetime
import json
import locale
import os
import platform
import re
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

# System Environment Information
SystemEnv = namedtuple(
    "SystemEnv",
    [
        "torch_version",
        "cuda_compiled_version",
        "gcc_version",
        "clang_version",
        "cmake_version",
        "os",
        "libc_version",
        "python_version",
        "python_platform",
        "is_cuda_available",
        "cuda_runtime_version",
        "cuda_module_loading",
        "nvidia_driver_version",
        "nvidia_gpu_models",
        "cudnn_version",
        "pip_version",  # 'pip' or 'pip3'
        "pip_packages",
        "conda_packages",
        "hip_compiled_version",
        "hip_runtime_version",
        "miopen_runtime_version",
        "caching_allocator_config",
        "is_xnnpack_available",
        "cpu_info",
    ],
)

COMMON_PATTERNS = [
    "torch",
    "numpy",
    "triton",
    "optree",
]

NVIDIA_PATTERNS = [
    "cuda-cudart",
    "cuda-cupti",
    "cuda-libraries",
    "cuda-opencl",
    "cuda-nvrtc",
    "cuda-runtime",
    "cublas",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nccl",
    "nvjitlink",
    "nvtx",
]

CONDA_PATTERNS = [
    "cudatoolkit",
    "soumith",
    "mkl",
    "magma",
]

PIP_PATTERNS = [
    "mypy",
    "flake8",
    "onnx",
]


def _run(command):
    """Return (return-code, stdout, stderr)."""
    shell = isinstance(command, str)
    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
    )  # nosec: Trusted input;
    # ensure command is not derived from user data.
    raw_output, raw_err = p.communicate()
    rc = p.returncode
    enc = "oem" if _get_platform() == "win32" else locale.getpreferredencoding()
    output = raw_output.decode(enc)
    err = raw_err.decode(enc)
    return rc, output.strip(), err.strip()


def _run_and_read_all(run_lambda, command):
    """Run command using run_lambda; reads and returns entire output if rc is 0."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def _run_and_parse_first_match(run_lambda, command, regex):
    """Run command using run_lambda, returns the first regex match if it exists."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def _run_and_return_first_line(run_lambda, command):
    """Run command using run_lambda and returns first line if output is not empty."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out.split("\n")[0]


def _get_conda_packages(run_lambda, patterns=None):
    if patterns is None:
        patterns = CONDA_PATTERNS + COMMON_PATTERNS + NVIDIA_PATTERNS
    conda = os.environ.get("CONDA_EXE", "conda")
    out = _run_and_read_all(run_lambda, f"{conda} list")
    if out is None:
        return out

    return "\n".join(
        line
        for line in out.splitlines()
        if not line.startswith("#") and any(name in line for name in patterns)
    )


def _get_gcc_version(run_lambda):
    return _run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")


def _get_clang_version(run_lambda):
    return _run_and_parse_first_match(
        run_lambda,
        "clang --version",
        r"clang version (.*)",
    )


def _get_cmake_version(run_lambda):
    return _run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")


def _get_nvidia_driver_version(run_lambda):
    if _get_platform() == "darwin":
        cmd = "kextstat | grep -i cuda"
        return _run_and_parse_first_match(
            run_lambda,
            cmd,
            r"com[.]nvidia[.]CUDA [(](.*?)[)]",
        )
    smi = _get_nvidia_smi()
    return _run_and_parse_first_match(run_lambda, smi, r"Driver Version: (.*?) ")


def _get_gpu_info(run_lambda):
    if _get_platform() == "darwin" or (
        TORCH_AVAILABLE
        and hasattr(torch.version, "hip")
        and torch.version.hip is not None
    ):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if torch.version.hip is not None:
                prop = torch.cuda.get_device_properties(0)
                if hasattr(prop, "gcnArchName"):
                    gcnArch = f" ({prop.gcnArchName})"
                else:
                    gcnArch = "NoGCNArchNameOnOldPyTorch"
            else:
                gcnArch = ""
            return torch.cuda.get_device_name(None) + gcnArch
        return None
    smi = _get_nvidia_smi()
    uuid_regex = re.compile(r" \(UUID: .+?\)")
    rc, out, _ = run_lambda(f"{smi} -L")
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return re.sub(uuid_regex, "", out)


def _get_running_cuda_version(run_lambda):
    return _run_and_parse_first_match(run_lambda, "nvcc --version", r"release .+ V(.*)")


def _get_cudnn_version(run_lambda):
    """Return a list of libcudnn.so; it's hard to tell which one is being used."""
    if _get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        cuda_path = os.environ.get("CUDA_PATH", "%CUDA_PATH%")
        where_cmd = str(Path(system_root) / "System32" / "where")
        cudnn_cmd = f'{where_cmd} /R "{cuda_path}\\bin" cudnn*.dll'
    elif _get_platform() == "darwin":
        # CUDA libraries and drivers can be found in /usr/local/cuda/.
        cudnn_cmd = "ls /usr/local/cuda/lib/libcudnn*"
    else:
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
    rc, out, _ = run_lambda(cudnn_cmd)
    # find will return 1 if there are permission errors or if not found
    if len(out) == 0 or (rc not in (1, 0)):
        cudnn_library = os.environ.get("CUDNN_LIBRARY")
        if cudnn_library is not None and Path(cudnn_library).is_file():
            return os.path.realpath(cudnn_library)
        return None
    files_set = set()
    for file_line in out.split("\n"):
        real_path = os.path.realpath(file_line)  # eliminate symbolic links
        if Path(real_path).is_file():
            files_set.add(real_path)
    if not files_set:
        return None
    files = sorted(files_set)
    if len(files) == 1:
        return files[0]
    result = "\n".join(files)
    return f"Probably one of the following:\n{result}"


def _get_nvidia_smi():
    # Note: nvidia-smi is currently available only on Windows and Linux
    smi = "nvidia-smi"
    if _get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        program_files_root = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        legacy_path = str(
            Path(program_files_root) / "NVIDIA Corporation" / "NVSMI" / smi,
        )
        new_path = str(Path(system_root) / "System32" / smi)
        smis = [new_path, legacy_path]
        for candidate_smi in smis:
            if Path(candidate_smi).exists():
                smi = f'"{candidate_smi}"'
                break
    return smi


def _get_cpu_info(run_lambda):
    rc, out, err = 0, "", ""
    if _get_platform() == "linux":
        rc, out, err = run_lambda("lscpu")
    elif _get_platform() == "win32":
        powershell_cmd = (
            'powershell.exe "gwmi -Class Win32_Processor '
            "| Select-Object -Property Name,Manufacturer,Family,Architecture,"
            "ProcessorType,DeviceID,CurrentClockSpeed,MaxClockSpeed,"
            'L2CacheSize,L2CacheSpeed,Revision | ConvertTo-Json"'
        )
        rc, out, err = run_lambda(powershell_cmd)
        if rc == 0:
            lst = []
            try:
                obj = json.loads(out)
                if isinstance(obj, list):
                    for o in obj:
                        lst.append("----------------------")
                        lst.extend(f"{k}: {v}" for k, v in o.items())
                else:
                    lst.extend(f"{k}: {v}" for k, v in obj.items())
            except ValueError as e:
                lst.append(out)
                lst.append(str(e))
            out = "\n".join(lst)
    elif _get_platform() == "darwin":
        rc, out, err = run_lambda("sysctl -n machdep.cpu.brand_string")
    return out if rc == 0 else err


def _get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("win32"):
        return "win32"
    if sys.platform.startswith("cygwin"):
        return "cygwin"
    if sys.platform.startswith("darwin"):
        return "darwin"
    return sys.platform


def _get_mac_version(run_lambda):
    return _run_and_parse_first_match(run_lambda, "sw_vers -productVersion", r"(.*)")


def _get_windows_version(run_lambda):
    powershell_cmd = (
        'powershell.exe "gwmi -Class Win32_OperatingSystem '
        "| Select-Object -Property Caption,OSArchitecture,Version "
        '| ConvertTo-Json"'
    )
    ret = _run_and_read_all(run_lambda, powershell_cmd)
    try:
        obj = json.loads(ret)
        ret = f'{obj["Caption"]} ({obj["Version"]} {obj["OSArchitecture"]})'
    except ValueError as e:
        ret += f"\n{e}"
    return ret


def _get_lsb_version(run_lambda):
    return _run_and_parse_first_match(
        run_lambda,
        "lsb_release -a",
        r"Description:\t(.*)",
    )


def _check_release_file(run_lambda):
    return _run_and_parse_first_match(
        run_lambda,
        "cat /etc/*-release",
        r'PRETTY_NAME="(.*)"',
    )


def _get_os(run_lambda):
    from platform import machine

    plat = _get_platform()
    os_description = None

    if plat in ("win32", "cygwin"):
        os_description = _get_windows_version(run_lambda)
    elif plat == "darwin":
        version = _get_mac_version(run_lambda)
        os_description = f"macOS {version} ({machine()})" if version else None
    elif plat == "linux":
        desc = _get_lsb_version(run_lambda)
        if desc is None:
            desc = _check_release_file(run_lambda)
        os_description = f"{desc} ({machine()})" if desc else f"{plat} ({machine()})"
    else:
        os_description = plat

    return os_description


def _get_python_platform():
    return platform.platform()


def _get_libc_version():
    if _get_platform() != "linux":
        return "N/A"
    return "-".join(platform.libc_ver())


def _get_pip_packages(run_lambda, patterns=None):
    """Return `pip list` output.

    Note: Will also find conda-installed pytorch and numpy packages.
    """
    if patterns is None:
        patterns = PIP_PATTERNS + COMMON_PATTERNS + NVIDIA_PATTERNS

    pip_version = "pip3" if sys.version_info[0] >= 3 else "pip"

    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    out = _run_and_read_all(
        run_lambda,
        [sys.executable, "-mpip", "list", "--format=freeze"],
    )
    filtered_out = "\n".join(
        line for line in out.splitlines() if any(name in line for name in patterns)
    )

    return pip_version, filtered_out


def _get_cachingallocator_config():
    return os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")


def _get_cuda_module_loading_config():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.init()
        return os.environ.get("CUDA_MODULE_LOADING", "")
    return "N/A"


def _is_xnnpack_available():
    if TORCH_AVAILABLE:
        import torch.backends.xnnpack

        return str(torch.backends.xnnpack.enabled)  # type: ignore[attr-defined]
    return "N/A"


def _get_env_info():
    """Collect environment information to aid in debugging.

    The returned environment information contains details on torch version,
    whether it is a debug build or not, the CUDA compiled version, gcc version,
    clang version, and cmake version, as well as the operating system and libc
    version. It also includes the python version, python platform, CUDA
    availability, CUDA runtime version, and CUDA module loading configuration.
    Further, it details the GPU model and configuration, Nvidia driver version,
    cuDNN version, pip version, and versions of relevant pip and conda packages.
    Additionally, HIP runtime version, MIOpen runtime version, caching allocator
    config, XNNPACK availability, and CPU information are provided.

    Returns:
        SystemEnv (namedtuple): A tuple containing various environment details
        and system information.
    """
    run_lambda = _run
    pip_version, pip_list_output = _get_pip_packages(run_lambda)
    if TORCH_AVAILABLE:
        version_str = torch.__version__
        cuda_available_str = str(torch.cuda.is_available())
        cuda_version_str = torch.version.cuda
        if (
            not hasattr(torch.version, "hip") or torch.version.hip is None
        ):  # cuda version
            hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"
        else:  # HIP version

            def get_version_or_na(cfg, prefix):
                _lst = [s.rsplit(None, 1)[-1] for s in cfg if prefix in s]
                return _lst[0] if _lst else "N/A"

            cfg = torch._C._show_config().split("\n")
            hip_runtime_version = get_version_or_na(cfg, "HIP Runtime")
            miopen_runtime_version = get_version_or_na(cfg, "MIOpen")
            cuda_version_str = "N/A"
            hip_compiled_version = torch.version.hip
    else:
        version_str = cuda_available_str = cuda_version_str = "N/A"
        hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"

    sys_version = sys.version.replace("\n", " ")
    conda_packages = _get_conda_packages(run_lambda)
    return SystemEnv(
        torch_version=version_str,
        python_version=f"{sys_version} ({sys.maxsize.bit_length() + 1}-bit runtime)",
        python_platform=_get_python_platform(),
        is_cuda_available=cuda_available_str,
        cuda_compiled_version=cuda_version_str,
        cuda_runtime_version=_get_running_cuda_version(run_lambda),
        cuda_module_loading=_get_cuda_module_loading_config(),
        nvidia_gpu_models=_get_gpu_info(run_lambda),
        nvidia_driver_version=_get_nvidia_driver_version(run_lambda),
        cudnn_version=_get_cudnn_version(run_lambda),
        hip_compiled_version=hip_compiled_version,
        hip_runtime_version=hip_runtime_version,
        miopen_runtime_version=miopen_runtime_version,
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=conda_packages,
        os=_get_os(run_lambda),
        libc_version=_get_libc_version(),
        gcc_version=_get_gcc_version(run_lambda),
        clang_version=_get_clang_version(run_lambda),
        cmake_version=_get_cmake_version(run_lambda),
        caching_allocator_config=_get_cachingallocator_config(),
        is_xnnpack_available=_is_xnnpack_available(),
        cpu_info=_get_cpu_info(run_lambda),
    )


env_info_fmt = """
PyTorch version: {torch_version}
CUDA used to build PyTorch: {cuda_compiled_version}
ROCM used to build PyTorch: {hip_compiled_version}

OS: {os}
GCC version: {gcc_version}
Clang version: {clang_version}
CMake version: {cmake_version}
Libc version: {libc_version}

Python version: {python_version}
Python platform: {python_platform}
Is CUDA available: {is_cuda_available}
CUDA runtime version: {cuda_runtime_version}
CUDA_MODULE_LOADING set to: {cuda_module_loading}
GPU models and configuration: {nvidia_gpu_models}
Nvidia driver version: {nvidia_driver_version}
cuDNN version: {cudnn_version}
HIP runtime version: {hip_runtime_version}
MIOpen runtime version: {miopen_runtime_version}
Is XNNPACK available: {is_xnnpack_available}

CPU:
{cpu_info}
""".strip()


def _replace_nones(dct, replacement="Could not collect"):
    for key in dct:
        if dct[key] is None:
            dct[key] = replacement
    return dct


def _replace_bools(dct, true="Yes", false="No"):
    for key in dct:
        if dct[key] is True:
            dct[key] = true
        elif dct[key] is False:
            dct[key] = false
    return dct


def _prepend(text, tag="[prepend]"):
    lines = text.split("\n")
    updated_lines = [f"{tag}{line}" for line in lines]
    return "\n".join(updated_lines)


def _replace_if_empty(text, replacement="No relevant packages"):
    if text is not None and len(text) == 0:
        return replacement
    return text


def _maybe_start_on_next_line(string):
    # If `string` is multiline, prepend a \n to it.
    if string is not None and len(string.split("\n")) > 1:
        return f"\n{string}\n"
    return string


def _pretty_str(envinfo):
    mutable_dict = envinfo._asdict()

    # If nvidia_gpu_models is multiline, start on the next line
    mutable_dict["nvidia_gpu_models"] = _maybe_start_on_next_line(
        envinfo.nvidia_gpu_models,
    )

    # If the machine doesn't have CUDA, report some fields as 'No CUDA'
    dynamic_cuda_fields = [
        "cuda_runtime_version",
        "nvidia_gpu_models",
        "nvidia_driver_version",
    ]
    all_cuda_fields = [*dynamic_cuda_fields, "cudnn_version"]
    all_dynamic_cuda_fields_missing = all(
        mutable_dict[field] is None for field in dynamic_cuda_fields
    )
    if (
        TORCH_AVAILABLE
        and not torch.cuda.is_available()
        and all_dynamic_cuda_fields_missing
    ):
        for field in all_cuda_fields:
            mutable_dict[field] = "No CUDA"
        if envinfo.cuda_compiled_version is None:
            mutable_dict["cuda_compiled_version"] = "None"

    # Replace True with Yes, False with No
    mutable_dict = _replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = _replace_nones(mutable_dict)

    # If either of these are '', replace with 'No relevant packages'
    mutable_dict["pip_packages"] = _replace_if_empty(mutable_dict["pip_packages"])
    mutable_dict["conda_packages"] = _replace_if_empty(mutable_dict["conda_packages"])

    # Tag conda and pip packages with a prefix
    # If they were previously None, they'll show up as ie '[conda] Could not collect'
    if mutable_dict["pip_packages"]:
        mutable_dict["pip_packages"] = _prepend(
            mutable_dict["pip_packages"],
            f"[{envinfo.pip_version}] ",
        )
    if mutable_dict["conda_packages"]:
        mutable_dict["conda_packages"] = _prepend(
            mutable_dict["conda_packages"],
            "[conda] ",
        )
    mutable_dict["cpu_info"] = envinfo.cpu_info
    return env_info_fmt.format(**mutable_dict)


def _get_pretty_env_info():
    """Return a pretty string of environment information.

    This function retrieves environment information by calling the `_get_env_info`
    function and then formats the information into a human-readable string.
    The retrieved environment information is listed in the documentation of
    `_get_env_info`. This function is used in `python collect_env.py` when reporting
    a bug.

    Returns:
        str: A pretty string of the environment information.
    """
    return _pretty_str(_get_env_info())


def _get_deps_info():
    """Overview of the installed version of main dependencies.

    Returns:
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    deps = [
        "torch",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "typing_extensions",
        "einops",
        "huggingface-hub",
    ]

    from importlib.metadata import version as get_version  # Rename to avoid shadowing

    try:
        __version__ = get_version("tabpfn")
    except ImportError:
        __version__ = "unknown"

    deps_info = {
        "tabpfn": __version__,
    }

    from importlib.metadata import PackageNotFoundError

    for modname in deps:
        try:
            deps_info[modname] = get_version(modname)  # Use renamed function
        except PackageNotFoundError:
            deps_info[modname] = "Not Found"  # Use a string instead of None
    return deps_info


def display_debug_info():
    """Print useful debugging information."""
    print("Collecting system and dependency information...")
    sys_info = _get_env_info()._replace(pip_packages="", conda_packages="")
    print(
        _pretty_str(sys_info).replace(
            "Versions of relevant libraries:\n"
            "[pip3] No relevant packages\n"
            "[conda] No relevant packages\n",
            "",
        )
    )

    # Print dependency versions in a clear section
    print("\nDependency Versions:")
    print("-" * 20)
    deps_info = _get_deps_info()
    for pkg, version in deps_info.items():
        print(f"{pkg}: {version or 'Not installed'}")

    # Check for crash dumps
    if (
        TORCH_AVAILABLE
        and hasattr(torch, "utils")
        and hasattr(torch.utils, "_crash_handler")
    ):
        minidump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR
        if sys.platform == "linux" and Path(minidump_dir).exists():
            dumps = [
                str(Path(minidump_dir) / dump) for dump in os.listdir(minidump_dir)
            ]
            if dumps:  # Only proceed if there are dumps
                latest = max(dumps, key=lambda p: Path(p).stat().st_ctime)
                ctime = Path(latest).stat().st_ctime
                crash_time = datetime.datetime.fromtimestamp(ctime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(f"\nLatest crash dump found from: {crash_time}")
                print(f"Location: {latest}")
