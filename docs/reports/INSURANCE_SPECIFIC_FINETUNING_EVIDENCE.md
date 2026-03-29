# Evidence Review: Insurance-Specific Fine-Tuning for TabPFN Classifier

Date: 2026-03-29

## Objective

Assess whether current project evidence supports the claim that fine-tuning a TabPFN classifier on insurance datasets would improve performance on insurance tasks.

This report distinguishes clearly between:
- **Measured evidence already available in this workspace**
- **Reasonable hypothesis supported indirectly by current findings**
- **Claims that are not yet proven**

---

## Short Answer

**Insurance-specific fine-tuning is a credible hypothesis, but it is not yet demonstrated by direct experimental evidence in this workspace.**

What we can support today is the following:
1. TabPFN already performs competitively on several insurance datasets without insurance-specific fine-tuning.
2. Some of the remaining weakness appears consistent with a **pre-training prior mismatch**, which makes domain-specific fine-tuning a plausible next step.
3. The current project has shown that **post-hoc calibration** can materially improve insurance performance, especially on probability accuracy.
4. However, no current artifact in this workspace shows a controlled **before-vs-after insurance fine-tuning experiment** for the classifier.

---

## 1) What Has Actually Been Measured

### A. TabPFN is already competitive on multiple insurance classification datasets

Current classification benchmarking across four insurance datasets shows:
- **ROC AUC wins for TabPFN on 3/4 datasets**
- **PR AUC wins for TabPFN on 2/4 datasets**

Observed datasets:
- EU Direct Lapse
- COIL 2000 (NL)
- Aus. Vehicle (AU)
- freMTPL2 Binary (FR)

### Interpretation
- TabPFN already transfers reasonably well to insurance tasks without a bespoke insurance fine-tune.
- This means insurance fine-tuning is **not required to make TabPFN viable** on insurance data.
- It also means that any future insurance-specific fine-tuning should be evaluated as an **incremental gain question**, not as a rescue strategy for a failing model.

---

### B. EU Direct Lapse shows a remaining gap that may be domain-related

In the multi-dataset head-to-head comparison, EU Direct Lapse is a notable exception:
- **GLM is better on ROC AUC**
- **TabPFN is only slightly better on PR AUC**

### Interpretation
- The problem is not that TabPFN fails uniformly on insurance data.
- The evidence instead suggests **dataset-specific mismatch**.
- EU Direct Lapse may be a case where the model's inherited priors or probability scaling are not ideally aligned with the structure of this particular insurance problem.

---

### C. Post-hoc calibration already improves insurance performance materially

For the lapse prediction workflow, post-hoc optimization produced a measurable lift:
- **Brier score improved by +1.66%**
- **ROC AUC improved by +0.0303**

Best observed variant:
- Engineered features + isotonic calibration

### Interpretation
- This is important because it shows the model's outputs can be improved on insurance data **without changing the underlying weights**.
- That supports the idea that at least part of the remaining error is due to **output calibration / prior mismatch**, not complete failure to learn relevant structure.
- If post-hoc correction helps, then true weight fine-tuning is at least conceptually plausible as a stronger form of domain adaptation.

---

## 2) Why Insurance-Specific Fine-Tuning Is a Plausible Hypothesis

The strongest indirect argument comes from the project's own error analysis.

### A. Evidence points to pre-training prior mismatch

Existing notes argue that TabPFN's calibration issue is likely inherited from pre-training rather than caused only by class imbalance in the insurance sample.

Observed reasoning in the project:
- Rebalancing the training data had little effect on TabPFN outputs.
- Probability compression remained even when the sample distribution changed.
- Isotonic calibration improved performance at the output layer.

### Why this matters

If the problem is truly **pre-training prior mismatch**, then fine-tuning on insurance datasets becomes a reasonable scientific hypothesis:

> Training the model further on insurance-like tasks may shift its internal priors and probability behavior closer to insurance-specific structure.

That could improve:
- Probability calibration
- Ranking quality on insurance targets
- Robustness across insurance datasets with similar covariate patterns

---

### B. Insurance data have recurring structure that a domain-specific fine-tune could exploit

Across the datasets used in this project, insurance problems often share patterns such as:
- Imbalanced binary outcomes
- Heterogeneous categorical and numeric features
- Premium- and exposure-related variables
- Behavioral and contractual drivers
- Stable but domain-specific nonlinear interactions

### Interpretation

If several training tasks come from the same broad domain, a domain-specific fine-tune could help the model adapt to:
- Typical insurance base rates
- Common feature semantics and scales
- Domain-relevant interaction patterns
- Insurance-style target noise characteristics

This is a valid motivation for an insurance fine-tuning experiment.

---

### C. Upstream TabPFN evidence shows finetuning can improve classifier performance in general

The upstream TabPFN project changed the default classifier model to a **finetuned checkpoint** specifically to improve out-of-the-box performance.

### Interpretation

- This does **not** prove that insurance-specific fine-tuning will help insurance tasks.
- But it does show that classifier fine-tuning is not merely theoretical; it has already been useful in the broader TabPFN development workflow.
- Therefore the proposed insurance-specific extension is technically credible.

---

## 3) What We Cannot Claim Yet

The following claims are **not yet supported by direct evidence** in this workspace:

1. **That insurance-specific fine-tuning improves TabPFN classifier performance on insurance datasets.**
2. **That any such improvement would be consistent across all insurance datasets.**
3. **That gains would occur on the most important metric for each task** (for example ROC AUC, PR AUC, Brier score, or log loss).
4. **That fine-tuning would outperform simpler post-hoc calibration methods on cost-benefit grounds.**

These claims remain unproven because no current artifact shows a controlled classifier experiment of the form:

- Baseline default TabPFN classifier
- Same train/validation/test split
- Same insurance dataset
- Same evaluation metrics
- Fine-tuned TabPFN classifier after training on insurance data
- Direct before-vs-after comparison

Without that experiment, the hypothesis remains plausible but unverified.

---

## 4) Current Best Interpretation

The most defensible interpretation of current evidence is:

1. **TabPFN already works reasonably well on insurance data.**
2. **Remaining weaknesses are consistent with domain/prior mismatch rather than total model inadequacy.**
3. **Because of that, insurance-specific fine-tuning is a credible next experiment.**
4. **However, the current evidence supports calibration-focused improvement more strongly than it supports true weight fine-tuning.**

In practical terms:
- The project currently has better evidence for **post-hoc calibration** than for **insurance-specific classifier fine-tuning**.
- Fine-tuning is best described as a **next-stage research hypothesis**, not a completed conclusion.

---

## 5) Recommended Report Language

Use wording like this when describing the current state of evidence:

> Current results do not yet prove that insurance-specific fine-tuning improves TabPFN classifier performance on insurance tasks. However, the observed pre-training prior mismatch, the dataset-specific weakness on EU Direct Lapse, and the measurable gains from post-hoc calibration together suggest that domain-specific fine-tuning is a credible next-step hypothesis, especially for improving probability calibration rather than raw discrimination.

Shorter version:

> Insurance-specific fine-tuning is plausible and technically well motivated, but not yet demonstrated by direct evidence in the current project artifacts.

---

## 6) What Evidence Would Be Needed to Prove It

To establish the claim properly, the next experiment should run a controlled insurance fine-tuning benchmark.

### Minimum experimental design

For each insurance classification dataset:
- Use a fixed train/validation/test split
- Evaluate the **default TabPFN classifier** before any additional training
- Fine-tune the classifier on insurance training data only
- Re-evaluate on the untouched test split
- Record:
  - ROC AUC
  - PR AUC
  - Brier score
  - Log loss
  - Calibration curve / ECE if available

### Stronger design

To test true domain adaptation rather than single-dataset overfitting:
- Fine-tune on a pool of insurance datasets
- Test on a held-out insurance dataset not used in fine-tuning

This would answer the more important question:

> Does insurance-specific fine-tuning improve generalization to insurance tasks as a domain, rather than only memorizing one dataset?

---

## 7) Practical Conclusion

### What is supported now
- TabPFN is already competitive on insurance classification tasks.
- The remaining lapse-model weakness appears consistent with prior/calibration mismatch.
- That makes insurance-specific fine-tuning a reasonable next experiment.

### What is not supported now
- A claim that insurance-specific fine-tuning has already been shown to improve classifier performance.

### Final position

**The project currently provides a strong rationale for trying insurance-specific fine-tuning, but not yet evidence that it works.**

That is the correct technical framing.

---

## 8) Evidence Layers Summary

### Direct evidence
- Multi-dataset TabPFN vs GLM insurance benchmarking
- EU Direct Lapse results
- Post-hoc calibration gains on insurance lapse prediction

### Indirect evidence
- Pre-training prior mismatch explanation
- Upstream use of a finetuned default classifier
- Shared structural characteristics across insurance datasets

### Not yet available
- Controlled before/after insurance-specific classifier fine-tuning results

---

## 9) Bottom-Line Statement

If a single-sentence conclusion is needed, use this:

> We do not yet have direct evidence that insurance-specific fine-tuning improves TabPFN classifier performance on insurance data, but the current results provide a clear technical rationale for testing it, particularly as a way to address domain-specific prior and calibration mismatch.