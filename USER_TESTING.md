# User Testing

## Install from this branch

```bash
pip install git+https://github.com/PriorLabs/TabPFN.git@brendan/hf-ungating
```

## Run the quick test

```bash
# simulate gating removal by setting a token
export HF_TOKEN=...
python tests/quick_test.py
```

## Reset login / cached models

To clear cached model weights and your auth token (e.g. to test a fresh login flow):

```bash
rm ~/Library/Caches/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt
rm ~/Library/Caches/tabpfn/tabpfn-v2.5-regressor-v2.5_default.ckpt
rm ~/.cache/tabpfn/auth_token
```

On Linux the model cache is typically at `~/.cache/tabpfn/` instead of `~/Library/Caches/tabpfn/`.
