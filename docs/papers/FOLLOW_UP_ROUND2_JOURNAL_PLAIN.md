# Plain‑English Summary — Follow‑Up to TabPFN Insurance Study

Key takeaways
- There is no single best model for all insurance problems; choose by the business goal.  
- TabPFN often improves ranking (who is more likely to have an event) but is not always better at predicting rare events precisely.  
- For numerical predictions, modern tree boosting models (e.g., CatBoost) usually perform best on noisy/count targets; TabPFN can be competitive on premium/value targets but needs careful confirmation.

Plain abstract (2 sentences)
This study compares a pretrained tabular model called TabPFN with standard logistic regression and common regression models across several insurance datasets. Results show TabPFN frequently helps with ranking customers by risk but is not a universal replacement; model choice should be driven by the specific business metric you care about.

What we did (short)
- Classification: Tested on four insurance datasets (EU Direct Lapse, COIL 2000, Australian vehicle, and freMTPL2) and compared a logistic regression (GLM) to TabPFN. We measured how well models rank risk and how well they do on rare events.  
- Regression: Tested on three insurance targets (claim frequency, premium, vehicle value) using simple baselines (linear, random forest, CatBoost) and compared typical error measures.
- We used consistent train/test splits and the same random seed and a capped training size so comparisons are fair.

How we measured things (very simple)
- ROC AUC: How well the model orders higher‑risk cases above lower‑risk ones (1.0 perfect, 0.5 random).  
- PR AUC: Focuses on performance for the rare, positive class (useful when events are uncommon).  
- MAE / RMSE: Average size of prediction errors for continuous targets.  
- R²: Roughly, how much of the variation in the target the model explains.

Main findings (plain)
- Classification: TabPFN had higher ROC AUC on three out of four datasets, so it often ranks customers better. For precision on rare events (PR AUC) the wins were mixed—TabPFN helped on some datasets but not all. The EU Direct Lapse dataset remained an exception where the GLM kept a ROC advantage.  
- Regression: Results depended on the target. CatBoost often gave the smallest errors for noisy or count targets, while RandomForest did best for the premium target in the current reproducible runs. Earlier TabPFN runs suggested it can do well on premium‑style targets, but those earlier results need rerunning to be confirmed.

What this means for actuaries
- Pick the model based on the KPI you care about (ranking vs probability accuracy vs precise prediction).  
- Include TabPFN as a candidate when ranking is important; verify on your KPI of interest.  
- Always check calibration (make predicted probabilities match observed rates) before using probabilities in pricing or provisioning.

Short practical note on calibration
We use a simple post‑training adjustment (isotonic calibration) on held‑out data to make predicted probabilities better match observed frequencies; this keeps the ordering the same but makes the percentage numbers more trustworthy.

If you like this tone, I can: (a) replace the original short journal draft with this plain version, (b) produce a combined file with both technical and plain sections, or (c) shorten further for a one‑paragraph press summary.
