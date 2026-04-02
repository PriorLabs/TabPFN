# Domain Fine-Tuning Logbook

## Run Batch 2026-04-02T00:30:53Z

### Configuration
- Stage: A
- Targets: eudirectlapse, coil2000, ausprivauto0405, freMTPL2freq_binary
- Seed: 42
- Target rows per run: 2500
- Domain pool rows per non-target dataset: 1000
- Device/context/steps: cpu/64/1
- Models: logistic_regression, random_forest, raw TabPFN, domain_finetuned TabPFN
- CatBoost: not available in environment

### Results (Raw vs Domain-Finetuned TabPFN)
| Target Dataset | Raw ROC AUC | Tuned ROC AUC | Raw PR AUC | Tuned PR AUC | Raw Brier | Tuned Brier | Raw LogLoss | Tuned LogLoss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eudirectlapse | 0.5763 | 0.5334 | 0.1676 | 0.1338 | 0.1113 | 0.1141 | 0.3833 | 0.3965 |
| coil2000 | 0.7566 | 0.5541 | 0.1735 | 0.0672 | 0.0539 | 0.0567 | 0.2059 | 0.2281 |
| ausprivauto0405 | 0.6486 | 0.5525 | 0.1177 | 0.1104 | 0.0626 | 0.0633 | 0.2421 | 0.2481 |
| freMTPL2freq_binary | 0.5412 | 0.5819 | 0.0596 | 0.0814 | 0.0483 | 0.0486 | 0.2014 | 0.2024 |

### Interpretation
- Domain-finetuned minus raw TabPFN macro deltas:
- ROC AUC: -0.0752
- PR AUC: -0.0314
- Brier: +0.0017
- LogLoss: +0.0106
- In this low-budget Stage A setup (single fine-tune step), domain fine-tuning did not improve aggregate primary endpoints.
- Improvement was observed on freMTPL2freq_binary, but three targets degraded.

### Observations
- Single-step domain adaptation appears too weak or misaligned for stable cross-dataset uplift.
- Raw TabPFN remains the stronger baseline for most targets under current settings.
- Fit-time profile remains practical for pilot iteration, but TabPFN arms dominate runtime cost.

### Comments
- Keep this batch as baseline evidence and proceed with controlled scale-up using the same split policy.
- Next run recommendation: steps 3 then 5 with unchanged context and seeds before changing other knobs.
- Keep writing notes directly through `--observations` and `--comments` on each Stage A command so interpretation stays tied to the exact run.
## Run 2026-04-02T00:37:03.052552+00:00

### Configuration
- Stage: A
- Target dataset: eudirectlapse
- Seed: 42
- Target rows: 800
- Pool rows per dataset: 400
- Device/context/steps: cpu/64/1

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.6566 | 0.2373 | 0.1110 | 0.3714 | 0.0403 |  |
| random_forest | 0.6110 | 0.1843 | 0.1116 | 0.3793 | 0.0329 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.5896 | 0.2138 | 0.1118 | 0.3821 | 0.0209 |  |
| domain_finetuned | 0.6365 | 0.2154 | 0.1111 | 0.3793 | 0.0180 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC +0.0469, PR AUC +0.0016, Brier -0.0006, LogLoss -0.0029.
- Primary calibration endpoints improved for domain fine-tuned TabPFN on this target.

### Observations
- First automated logbook smoke-check after enabling markdown logging.

### Comments
- Confirm file append format and interpretation block; keep this as a process validation run.

## Run 2026-04-02T00:41:21.342091+00:00

### Configuration
- Stage: A
- Target dataset: eudirectlapse
- Seed: 42
- Target rows: 2500
- Pool rows per dataset: 1000
- Device/context/steps: cpu/64/3

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.6070 | 0.1944 | 0.1108 | 0.3818 | 0.0233 |  |
| random_forest | 0.5656 | 0.1702 | 0.1121 | 0.3849 | 0.0220 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.5763 | 0.1676 | 0.1113 | 0.3833 | 0.0201 |  |
| domain_finetuned | 0.5339 | 0.1338 | 0.1141 | 0.3970 | 0.0432 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC -0.0424, PR AUC -0.0337, Brier +0.0028, LogLoss +0.0138.
- Primary calibration endpoints degraded for domain fine-tuned TabPFN on this target.

### Observations
- Scale-up run with max_finetune_steps=3 under fixed Stage A protocol.

### Comments
- Step-3 sweep to test whether domain-finetuned TabPFN improves versus raw baseline.

## Run 2026-04-02T00:45:38.621798+00:00

### Configuration
- Stage: A
- Target dataset: coil2000
- Seed: 42
- Target rows: 2500
- Pool rows per dataset: 1000
- Device/context/steps: cpu/64/3

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.7428 | 0.1469 | 0.0554 | 0.2094 | 0.0114 |  |
| random_forest | 0.6868 | 0.1270 | 0.0587 | 0.2611 | 0.0221 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.7566 | 0.1735 | 0.0539 | 0.2059 | 0.0131 |  |
| domain_finetuned | 0.5532 | 0.0673 | 0.0566 | 0.2276 | 0.0076 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC -0.2034, PR AUC -0.1062, Brier +0.0027, LogLoss +0.0216.
- Primary calibration endpoints degraded for domain fine-tuned TabPFN on this target.

### Observations
- Scale-up run with max_finetune_steps=3 under fixed Stage A protocol.

### Comments
- Step-3 sweep to test whether domain-finetuned TabPFN improves versus raw baseline.

## Run 2026-04-02T00:47:01.290125+00:00

### Configuration
- Stage: A
- Target dataset: ausprivauto0405
- Seed: 42
- Target rows: 2500
- Pool rows per dataset: 1000
- Device/context/steps: cpu/64/3

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.6125 | 0.1084 | 0.0632 | 0.2448 | 0.0072 |  |
| random_forest | 0.5364 | 0.0779 | 0.0739 | 0.3439 | 0.0646 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.6486 | 0.1177 | 0.0626 | 0.2421 | 0.0173 |  |
| domain_finetuned | 0.5505 | 0.1102 | 0.0634 | 0.2483 | 0.0178 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC -0.0981, PR AUC -0.0075, Brier +0.0008, LogLoss +0.0062.
- Primary calibration endpoints degraded for domain fine-tuned TabPFN on this target.

### Observations
- Scale-up run with max_finetune_steps=3 under fixed Stage A protocol.

### Comments
- Step-3 sweep to test whether domain-finetuned TabPFN improves versus raw baseline.

## Run 2026-04-02T00:55:36.595308+00:00

### Configuration
- Stage: A
- Target dataset: freMTPL2freq_binary
- Seed: 42
- Target rows: 2500
- Pool rows per dataset: 1000
- Device/context/steps: cpu/64/3

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.5132 | 0.0551 | 0.0506 | 0.2193 | 0.0233 |  |
| random_forest | 0.4923 | 0.0520 | 0.0508 | 0.2460 | 0.0258 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.5412 | 0.0596 | 0.0483 | 0.2014 | 0.0048 |  |
| domain_finetuned | 0.5888 | 0.0814 | 0.0484 | 0.2007 | 0.0114 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC +0.0476, PR AUC +0.0217, Brier +0.0001, LogLoss -0.0007.
- Mixed calibration endpoint movement; keep this target in follow-up runs.

### Observations
- Scale-up run with max_finetune_steps=3 under fixed Stage A protocol.

### Comments
- Step-3 sweep to test whether domain-finetuned TabPFN improves versus raw baseline.

## Run 2026-04-02T00:58:57.660316+00:00

### Configuration
- Stage: A
- Target dataset: eudirectlapse
- Seed: 42
- Target rows: 2500
- Pool rows per dataset: 1000
- Device/context/steps: cpu/64/5

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.6070 | 0.1944 | 0.1108 | 0.3818 | 0.0233 |  |
| random_forest | 0.5656 | 0.1702 | 0.1121 | 0.3849 | 0.0220 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.5763 | 0.1676 | 0.1113 | 0.3833 | 0.0201 |  |
| domain_finetuned | 0.5337 | 0.1338 | 0.1143 | 0.3983 | 0.0450 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC -0.0426, PR AUC -0.0337, Brier +0.0030, LogLoss +0.0150.
- Primary calibration endpoints degraded for domain fine-tuned TabPFN on this target.

### Observations
- Scale-up run with max_finetune_steps=5 under fixed Stage A protocol.

### Comments
- Step-5 sweep to evaluate whether additional adaptation improves primary endpoints.

## Run 2026-04-02T01:03:23.689820+00:00

### Configuration
- Stage: A
- Target dataset: coil2000
- Seed: 42
- Target rows: 2500
- Pool rows per dataset: 1000
- Device/context/steps: cpu/64/5

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.7428 | 0.1469 | 0.0554 | 0.2094 | 0.0114 |  |
| random_forest | 0.6868 | 0.1270 | 0.0587 | 0.2611 | 0.0221 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.7566 | 0.1735 | 0.0539 | 0.2059 | 0.0131 |  |
| domain_finetuned | 0.5527 | 0.0671 | 0.0566 | 0.2278 | 0.0091 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC -0.2039, PR AUC -0.1064, Brier +0.0027, LogLoss +0.0219.
- Primary calibration endpoints degraded for domain fine-tuned TabPFN on this target.

### Observations
- Scale-up run with max_finetune_steps=5 under fixed Stage A protocol.

### Comments
- Step-5 sweep to evaluate whether additional adaptation improves primary endpoints.

## Run 2026-04-02T01:04:51.158499+00:00

### Configuration
- Stage: A
- Target dataset: ausprivauto0405
- Seed: 42
- Target rows: 2500
- Pool rows per dataset: 1000
- Device/context/steps: cpu/64/5

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.6125 | 0.1084 | 0.0632 | 0.2448 | 0.0072 |  |
| random_forest | 0.5364 | 0.0779 | 0.0739 | 0.3439 | 0.0646 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.6486 | 0.1177 | 0.0626 | 0.2421 | 0.0173 |  |
| domain_finetuned | 0.5508 | 0.1116 | 0.0633 | 0.2480 | 0.0171 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC -0.0978, PR AUC -0.0061, Brier +0.0007, LogLoss +0.0059.
- Primary calibration endpoints degraded for domain fine-tuned TabPFN on this target.

### Observations
- Scale-up run with max_finetune_steps=5 under fixed Stage A protocol.

### Comments
- Step-5 sweep to evaluate whether additional adaptation improves primary endpoints.

## Run 2026-04-02T01:12:38.207680+00:00

### Configuration
- Stage: A
- Target dataset: freMTPL2freq_binary
- Seed: 42
- Target rows: 2500
- Pool rows per dataset: 1000
- Device/context/steps: cpu/64/5

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.5132 | 0.0551 | 0.0506 | 0.2193 | 0.0233 |  |
| random_forest | 0.4923 | 0.0520 | 0.0508 | 0.2460 | 0.0258 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.5412 | 0.0596 | 0.0483 | 0.2014 | 0.0048 |  |
| domain_finetuned | 0.5909 | 0.0809 | 0.0485 | 0.2015 | 0.0148 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC +0.0496, PR AUC +0.0213, Brier +0.0002, LogLoss +0.0001.
- Primary calibration endpoints degraded for domain fine-tuned TabPFN on this target.

### Observations
- Scale-up run with max_finetune_steps=5 under fixed Stage A protocol.

### Comments
- Step-5 sweep to evaluate whether additional adaptation improves primary endpoints.

## Run 2026-04-02T01:23:20.961405+00:00

### Configuration
- Stage: A
- Target dataset: eudirectlapse
- Seed: 42
- Target rows: 2500
- Pool rows per dataset: 1000
- Device/context/steps: cpu/128/5

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.6070 | 0.1944 | 0.1108 | 0.3818 | 0.0233 |  |
| random_forest | 0.5656 | 0.1702 | 0.1121 | 0.3849 | 0.0220 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.5763 | 0.1676 | 0.1113 | 0.3833 | 0.0201 |  |
| domain_finetuned | 0.5218 | 0.1386 | 0.1125 | 0.3881 | 0.0279 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC -0.0545, PR AUC -0.0290, Brier +0.0012, LogLoss +0.0048.
- Primary calibration endpoints degraded for domain fine-tuned TabPFN on this target.

### Observations
- Context scale-up run: increased context samples to 128 with fixed Stage A setup.

### Comments
- Primary objective: test whether higher context unlocks domain-finetuned gains versus raw TabPFN.

## Run 2026-04-02T01:28:20.878603+00:00

### Configuration
- Stage: A
- Target dataset: coil2000
- Seed: 42
- Target rows: 2500
- Pool rows per dataset: 1000
- Device/context/steps: cpu/128/5

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.7428 | 0.1469 | 0.0554 | 0.2094 | 0.0114 |  |
| random_forest | 0.6868 | 0.1270 | 0.0587 | 0.2611 | 0.0221 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.7566 | 0.1735 | 0.0539 | 0.2059 | 0.0131 |  |
| domain_finetuned | 0.6037 | 0.0839 | 0.0562 | 0.2243 | 0.0050 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC -0.1529, PR AUC -0.0896, Brier +0.0023, LogLoss +0.0183.
- Primary calibration endpoints degraded for domain fine-tuned TabPFN on this target.

### Observations
- Context scale-up run: increased context samples to 128 with fixed Stage A setup.

### Comments
- Primary objective: test whether higher context unlocks domain-finetuned gains versus raw TabPFN.

## Run 2026-04-02T01:29:54.960871+00:00

### Configuration
- Stage: A
- Target dataset: ausprivauto0405
- Seed: 42
- Target rows: 2500
- Pool rows per dataset: 1000
- Device/context/steps: cpu/128/5

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.6125 | 0.1084 | 0.0632 | 0.2448 | 0.0072 |  |
| random_forest | 0.5364 | 0.0779 | 0.0739 | 0.3439 | 0.0646 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.6486 | 0.1177 | 0.0626 | 0.2421 | 0.0173 |  |
| domain_finetuned | 0.6093 | 0.1213 | 0.0630 | 0.2461 | 0.0189 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC -0.0393, PR AUC +0.0036, Brier +0.0004, LogLoss +0.0040.
- Primary calibration endpoints degraded for domain fine-tuned TabPFN on this target.

### Observations
- Context scale-up run: increased context samples to 128 with fixed Stage A setup.

### Comments
- Primary objective: test whether higher context unlocks domain-finetuned gains versus raw TabPFN.

## Run 2026-04-02T01:37:31.657107+00:00

### Configuration
- Stage: A
- Target dataset: freMTPL2freq_binary
- Seed: 42
- Target rows: 2500
- Pool rows per dataset: 1000
- Device/context/steps: cpu/128/5

### Results
| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.5132 | 0.0551 | 0.0506 | 0.2193 | 0.0233 |  |
| random_forest | 0.4923 | 0.0520 | 0.0508 | 0.2460 | 0.0258 |  |
| catboost |  |  |  |  |  | catboost_not_available: No module named 'catboost' |
| raw | 0.5412 | 0.0596 | 0.0483 | 0.2014 | 0.0048 |  |
| domain_finetuned | 0.5536 | 0.0708 | 0.0480 | 0.1999 | 0.0009 |  |

### Interpretation
- Domain-finetuned minus raw TabPFN: ROC AUC +0.0123, PR AUC +0.0112, Brier -0.0003, LogLoss -0.0015.
- Primary calibration endpoints improved for domain fine-tuned TabPFN on this target.

### Observations
- Context scale-up run: increased context samples to 128 with fixed Stage A setup.

### Comments
- Primary objective: test whether higher context unlocks domain-finetuned gains versus raw TabPFN.

