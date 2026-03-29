# Cleanup Plan: Remove Duplicate TabPFN Files

## What will be REMOVED from TabPFN-work-scott/
These are duplicates of the original repo - they'll stay in TabPFN-upstream/

```
src/                          # Original TabPFN source code
examples/                     # Original examples  
scripts/                      # Original scripts
tests/                        # Original tests (except your custom ones)
pyproject.toml               # Original config
CHANGELOG.md                 # Original docs
LICENSE                      # Original license
README.md                    # Original readme
```

## What will be KEPT in TabPFN-work-scott/
All your custom work:

```
ADSWP Project/               # Your domain-specific applications
├── TabPFN Classifier on eudirectlapse.ipynb
├── TabPFN_ausprivauto0405.R
├── TabPFN_freMTPL.R
├── TabPFN_on_freMTPL.ipynb
├── eudirectlapse.csv
└── baselining/              # Your experimental baselining work
    ├── Any custom notebooks/scripts you created
    └── Your results/outputs

BaselineExperiments/         # Your experimental framework
├── *.ipynb                  # Your notebooks
├── *.py                     # Your Python scripts
├── *.md                     # Your documentation
└── model_output.csv         # Your results

.git/                        # Git history (needed for your fork)
.github/                     # GitHub configurations
```

## File Count Summary
- Files to REMOVE: ~255 (original TabPFN files)
- Files to KEEP: ~321 (your custom work)
- .venv/ and other local files: Keep as needed

## Next Steps
1. Review this plan
2. Then I'll execute the cleanup by removing the duplicate folders
