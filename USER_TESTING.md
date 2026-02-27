# User Testing

Instructions for testing the Hugging Face ungating changes

## 1. Create a virtual environment

```bash
mkdir tabpfn-test && cd tabpfn-test
```

**With pip:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**With uv:**

```bash
uv venv .venv
source .venv/bin/activate
```

## 2. Install from this branch

**With pip:**

```bash
pip install "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git@brendan/hf-ungating"
```

**With uv:**

```bash
uv pip install "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git@brendan/hf-ungating"
```

## 3. Set environment variables

During testing TabPFN still requires a Hugging Face token for gated model access. Set `HF_TOKEN` before running anything.

**In a terminal (bash / zsh):**

```bash
export HF_TOKEN="hf_..."
```

**In a Jupyter notebook:**

```
%env HF_TOKEN=hf_...
```

> `%env` is an IPython magic that sets the variable for the current kernel process.
> For persistence across terminal sessions, add the `export` line to your `~/.bashrc` or `~/.zshrc`.

## 4. Run the quick test

Download and run the test script:

```bash
curl -L -o quick_test.py https://raw.githubusercontent.com/PriorLabs/TabPFN/brendan/hf-ungating/tests/quick_test.py
python quick_test.py
```

**Expected output:** the script runs a small classification and regression task and prints predictions for each. You should see arrays of predicted class labels (classification) and predicted values (regression) with no errors.

## 5. Find the model cache directory

To check where TabPFN stores downloaded models:

**From the terminal:**

```bash
python -c "from tabpfn.model_loading import get_cache_dir; print(get_cache_dir())"
```

**In a notebook:**

```python
from tabpfn.model_loading import get_cache_dir
print(get_cache_dir())
```

Platform defaults:

| Platform | Default location |
|----------|-----------------|
| macOS    | `~/Library/Caches/tabpfn/` |
| Linux    | `~/.cache/tabpfn/` |

## 6. Reset login / cached models

To clear cached model weights and your auth token (e.g. to test a fresh login flow):

```bash
# Find your cache directory:
python -c "from tabpfn.model_loading import get_cache_dir; print(get_cache_dir())"

# Then remove cached models and the auth token from that directory:
rm <cache_dir>/tabpfn-v2.5-classifier-v2.5_default.ckpt
rm <cache_dir>/tabpfn-v2.5-regressor-v2.5_default.ckpt
rm <cache_dir>/auth_token
```

Replace `<cache_dir>` with the path printed above (e.g. `~/.cache/tabpfn/` on Linux, `~/Library/Caches/tabpfn/` on macOS).
