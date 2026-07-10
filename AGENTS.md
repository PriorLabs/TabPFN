# AGENTS.md

Notes for agents/automation running TabPFN, especially on Turing-class
datacenter GPUs (e.g. Tesla T4, sm_75). None of this changes default behavior —
it documents what the code already does, verified on real hardware.

## GPU detection can silently fall back to CPU

`device="auto"` (the default) resolves via `torch.cuda.is_available()`. If that
is `False` — most commonly because `pip install tabpfn` pulled in a PyTorch
build compiled against a newer CUDA toolkit than your NVIDIA driver supports —
TabPFN returns a CPU device **with no warning or error**. There is no log line
telling you inference is running on CPU instead of GPU.

If you expect GPU inference, verify it explicitly after installing:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If this prints `False`, install a PyTorch build matching your driver's CUDA
version from the [PyTorch installation selector](https://pytorch.org/get-started/locally/)
before installing TabPFN. Example, for a driver that only supports CUDA 12.4
(e.g. driver `550.163.01`, common on T4/A10G cloud and Colab instances):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install tabpfn
```

## Autocast dtype on Turing (sm_75) is fp16, and that's correct

`inference_precision="auto"` enables autocast without specifying a dtype, so
CUDA's autocast defaults to `torch.float16`. Verified at runtime on a T4
(SDPA input tensors were `torch.float16`). This is the right choice for
Turing — the architecture has no bf16 tensor cores, so forcing bf16 would run
substantially slower. Do not override this to bf16 on T4/Turing hardware.

(The `# bfloat16` comment next to `AUTOCAST_DTYPE_BYTE_SIZE` in
`src/tabpfn/constants.py` refers to the byte size of the dtype — 2 bytes,
shared by fp16 and bf16 — not the actual dtype used at inference. Don't read
it as "autocast defaults to bf16".)

## SDPA attention backend on sm_75

TabPFN's SDPA backend list is `[FLASH, EFFICIENT, CUDNN, MATH]`, tried in
order. On sm_75 (Turing/T4), FlashAttention and cuDNN kernels are unavailable
(`RuntimeError: No available kernel`); the first eligible backend is
`EFFICIENT` (memory-efficient attention), with `MATH` as the final fallback.
This is already handled by the repo — no configuration needed — and was
confirmed by forcing each backend explicitly on a T4.

## v2 sample-count limit vs. OOM

TabPFN-2 rejects fits with more than 10,000 samples via
`TabPFNValidationError` rather than running out of memory. On a Tesla T4
(16GB), measured peak VRAM stayed well under the 16GB budget across the
supported range:

| Dataset | Peak VRAM | fit / predict |
|---|---|---|
| breast-cancer (569×30) | 0.15 GB | 0.42s / 0.41s |
| synthetic 2,000×50 | 0.59 GB | 0.70s / 3.57s |
| synthetic 10,000×100 | 5.33 GB | 3.49s / 84.3s |
| synthetic 24,000×200 | — | rejected: `TabPFNValidationError` (>10,000 samples) |

Environment: Tesla T4 16GB, driver 550.163.01 (CUDA 12.4 runtime), Python
3.10.12, `torch==2.6.0+cu124`, `tabpfn==8.0.8`. Measured via the repo's own
`fit`/`predict_proba` API (`create_default_for_version(ModelVersion.V2)`,
`load_breast_cancer` / synthetic data), no code changes.
