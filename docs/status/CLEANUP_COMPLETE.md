# ✓ Cleanup Complete: Repository Separated

## What Was Done

Your TabPFN-work-scott repository has been successfully cleaned up to **separate your custom work from the original TabPFN code**.

### Removed from TabPFN-work-scott/
All original/duplicated files are now **only in TabPFN-upstream**:
- `src/` — TabPFN source code
- `examples/` — Original examples
- `scripts/` — Original scripts  
- `tests/` — Original tests
- `ADSWP Project/baselining/tabpfn/` — Duplicate TabPFN copy
- `ADSWP Project/baselining/tabpfn-extensions/` — Extensions copy
- Root config files: `pyproject.toml`, `CHANGELOG.md`, `LICENSE`, `README.md`

### Kept in TabPFN-work-scott/
All your custom work is preserved:

#### 1. **ADSWP Project/**
Your domain-specific applications:
- `TabPFN Classifier on eudirectlapse.ipynb` — Analysis on eudirectlapse dataset
- `TabPFN_ausprivauto0405.R` — R analysis script
- `TabPFN_freMTPL.R` — R analysis for freMTPL
- `TabPFN_on_freMTPL.ipynb` — Jupyter notebook for freMTPL
- `eudirectlapse.csv` — Your dataset
- `baselining/` — Your experimental work folder

#### 2. **BaselineExperiments/**
Your experimental framework:
- ~10 Jupyter notebooks (training, evaluation, exploration)
- ~10 Python scripts (data loading, model training, evaluation)
- ~20 documentation files (guides, summaries, reports)
- Model outputs and results
- Data directories

---

## Next Steps

### For Reference to Original Code
When you need to reference the original TabPFN code or examples, use:
```
/Users/Scott/Documents/Data Science/ADSWP/TabPFN-upstream/
```

### For Your Work
Continue working in:
```
/Users/Scott/Documents/Data Science/ADSWP/TabPFN-work-scott/
```

### Git Tracking
Run this to commit the cleanup to your fork:
```bash
cd TabPFN-work-scott
git add -A
git commit -m "refactor: separate custom work from original TabPFN code"
git push
```

---

## Summary
- ✓ Duplicate files removed
- ✓ Custom work preserved  
- ✓ Clear separation between your fork and upstream
- ✓ Easier to sync with upstream in the future
