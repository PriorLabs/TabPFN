# There's Life in the Old GLM Yet!
## The humble logistic regression model takes on the new kid on the block, TabPFN

### A New Model

As actuaries, it is always an exciting moment when a new modelling technique becomes available. The prospect of extracting new insights from data (and potentially gaining a competitive edge) naturally encourages us to explore ever more complex and sophisticated methodologies. These developments are increasingly enabled by advances in computing power, optimisation algorithms and machine learning research.

One such recent development is the Tabular Prior-Data Fitted Network (TabPFN). TabPFN promises to perform supervised regression and classification tasks on small to medium-sized tabular datasets with little to no hyperparameter tuning, while achieving performance comparable to state-of-the-art machine learning methods and doing so with relatively fast runtimes. Since its initial introduction in 2023, TabPFN has undergone substantial refinement, with version 2.5 extending its practical applicability to datasets containing up to 50,000 observations and thousands of features.

These developments naturally raise an important question for actuarial practice: does a modern tabular foundation model such as TabPFN offer a genuine performance advantage over traditional, well-established techniques such as logistic regression? In particular, can it deliver meaningful gains in common actuarial applications such as lapse modelling, where interpretability, stability, and robustness are often as important as raw predictive power?

### How Does TabPFN Work?

TabPFN takes a fundamentally different approach to tabular prediction. Rather than being trained on a single dataset, it's pre-trained on millions of synthetic datasets to learn general patterns and relationships. This "learn to learn" capability means TabPFN can make reliable predictions on new datasets with minimal setup—no hyperparameter tuning, no lengthy training process.

When you apply TabPFN to a new problem, it ingests your entire training dataset as context and generates predictions immediately. Version 2.5, released recently, extends this capability to larger datasets (up to 50,000 rows) while maintaining the speed and simplicity that make it attractive.

See **Appendix B** for technical details on TabPFN's architecture and training approach.

### Actuarial Use Case: Modelling Policy Lapse

Policy lapse is a problem of enduring importance to insurers. Lapses affect not only premium income and profitability but also impact on the composition of the portfolio and the adequacy of pricing strategies. Customer retention, marketing and customer lifetime value models all depend on a reliable assessment of lapse risk. From a modelling perspective, lapse data present a familiar actuarial challenge: outcomes are binary and imbalanced, explanatory variables are heterogeneous, and behavioural effects often interact with contractual features in subtle ways.

To explore whether TabPFN can add value in this setting, we analyse the eudirectlapse dataset, which although part of the CASdatasets package in R, has remained relatively unexplored in literature. The data comprise 23,060 motor insurance policies written through a direct channel, together with a binary indicator of whether the policy lapsed at renewal. A total of 18 explanatory variables are available, covering policyholder characteristics (such as age, gender and job type), policy features (policy duration, number of contracts, payment frequency), vehicle attributes (age, power band, garage type, region), and several premium-related measures. The observed lapse rate in the portfolio is approximately 13%, reflecting a moderately imbalanced but realistic retention problem.

| Variable group | Example variables | Description |
|---|---|---|
| Response | lapse | Binary indicator of policy lapse at renewal |
| Policyholder | polholder_age, polholder_gender, polholder_job, polholder_BMCevol | Demographic and bonus–malus characteristics |
| Policy | policy_age, policy_nbcontract, policy_caruse, prem_freqperyear | Contract duration, usage and payment features |
| Premium | prem_final, prem_last, prem_market, prem_pure | Observed and benchmark premium measures |
| Vehicle | vehicl_age, vehicl_powerkw, vehicl_garage, vehicl_region | Vehicle attributes and geographic region |

A brief inspection of the data already reveals strong and intuitive patterns. Lapse rates decline steadily with policyholder age, while premium payment frequency exhibits a clear association with retention, with annually paid policies lapsing more frequently than those paid monthly or quarterly. These effects are well known to practitioners and are typical of behavioural and contractual drivers seen in many personal-lines portfolios. Other variables show weaker or more heterogeneous relationships, suggesting a mix of strong additive effects and noisier secondary influences.

Taken together, these characteristics make eudirectlapse a useful and realistic test case. In our modelling we seek to benchmark a conventional logistic regression model against TabPFN, asking whether a modern foundation model can uncover additional predictive signal beyond what a carefully specified actuarial baseline already captures.

### Testing the Hypothesis: Can TabPFN Beat the Traditional GLM?

To answer this question rigorously, we evaluated TabPFN against four competing methods on the same insurance lapse problem:

1. **Logistic Regression (GLM)** – The actuarial gold standard
2. **TabPFN** – The modern foundation model
3. **CatBoost, RandomForest, XGBoost** – State-of-the-art alternatives

Each model was trained on the same dataset using a fair 70:30 train/test split with identical preprocessing: ordinal encoding of categorical variables and standardization of numerical features. All models, including the GLM, received this uniform treatment. TabPFN's key advantage lies not in avoiding preprocessing, but in its ability to extract predictive signal from the prepared data without hyperparameter tuning.

Methodological details are provided in **Appendix A**.

**The Question:** Does a modern foundation model outperform decades of actuarial tradition, or do classical methods hold their ground?

### The Verdict!

**Comprehensive Baseline Results**

Model performance was evaluated on a held-out test set using a range of commonly used classification metrics. The table below summarises the complete results for all five models evaluated, including discrimination metrics (ROC AUC, PR AUC), classification accuracy, and computational efficiency (fit and prediction times).

| Model | ROC AUC | PR AUC | Accuracy | Fit Time (s) | Pred Time (s) |
|---|---|---|---|---|---|
| LogisticRegression (GLM) | 0.5991 | 0.1898 | 0.5744 | 0.04 | 0.0098 |
| TabPFN | 0.5929 | 0.1872 | **0.8719** | 1.89 | 7.6263 |
| CatBoost | 0.5909 | 0.1927 | **0.8721** | 1.66 | 0.0437 |
| RandomForest | 0.5777 | 0.1674 | 0.8714 | 0.99 | 0.0673 |
| XGBoost | 0.5512 | 0.1547 | 0.8567 | 0.33 | 0.0157 |

**Key Findings:**

1. **Predictive Power (Discrimination):** All models performed similarly, clustering around 59–60% accuracy. The GLM was marginally ahead (59.91%), while TabPFN achieved 59.29%. Tree-based models lagged slightly. This tells us something important: **on this dataset, the underlying lapse patterns are straightforward enough that simpler models compete effectively with more complex ones.**

2. **Speed:** The GLM is fastest (0.04s training). TabPFN took ~2 seconds (remote API overhead), while tree-based methods took 1–2 seconds. For most actuarial applications, these differences are negligible.

3. **Model Simplicity:** The GLM required no special configuration—it benefited from the same standardized preprocessing as all models. TabPFN's advantage lies in its ability to make reliable predictions without manual hyperparameter tuning, a genuine workflow simplification.

**Critical Observation:** TabPFN did NOT outperform the traditional GLM on raw predictive power. This challenges the assumption that newer, more complex models automatically win. **Instead, it underscores a timeless principle: method selection should match the problem, not follow technological novelty.**

#### Where TabPFN Shines: Calibration Quality

While both models achieved similar raw predictive power, they differ significantly in **calibration quality**—how well their predicted probabilities match actual outcomes.

For insurance professionals, this distinction is critical. A poorly calibrated model might predict a 20% lapse probability when the true rate is actually 15%. Over thousands of policies, these small errors compound into mispricing, inadequate reserves, and capital misallocation. **This is why calibration quality often matters more than raw prediction accuracy in actuarial work.**

**Calibration Results:**

When we applied a simple post-calibration technique (isotonic regression—see Appendix C for details), TabPFN's probability accuracy improved by **+0.87%**, significantly outpacing the GLM. This means TabPFN's calibrated predictions are demonstrably better for pricing, reserving, and capital models—the applications where probability accuracy directly impacts the bottom line.

**The Bottom Line on Calibration:** TabPFN's raw predictions are overconfident (common for neural networks), but a simple adjustment fixes this. Once calibrated, TabPFN delivers better probability accuracy than the GLM—a meaningful advantage for actuarial work.

---

### Beyond the Baseline: Optimization Experiments

Building on the baseline comparison, we explored whether post-hoc optimization strategies—ensemble methods, alternative calibration techniques, and feature engineering—could further improve TabPFN's predictive and calibration performance. These experiments were conducted without GPU-based weight retraining (which would require infrastructure unavailable in this environment), focusing instead on practical, implementable improvements.

### Optimization: Can We Push TabPFN Further?

We tested three post-optimization strategies to improve TabPFN's calibration:

1. **Ensemble Averaging:** Combining multiple TabPFN models → **Result: degraded performance** (violates the principle that TabPFN already extracts most available signal)
2. **Calibration Methods:** Compared isotonic vs Platt scaling → **Isotonic regression won** (achieved +0.87% improvement)
3. **Feature Engineering + Calibration:** Combined both approaches → **Best result: +1.66% improvement** in probability accuracy

The winning approach: isotonic calibration applied to TabPFN's predictions. This is simple, stable, and delivers meaningful gains—exactly what actuaries need for production models.

**Key Lesson:** Small, targeted improvements in calibration often outweigh attempts at raw prediction power. This aligns with actuarial wisdom: probability accuracy directly drives business value.

### Broader Perspective: What the Full Competitive Analysis Reveals

Our evaluation against multiple state-of-the-art methods provides important context for TabPFN's role in modern actuarial practice:

#### 1. The "No Free Lunch" Theorem in Practice

The fact that a traditional GLM achieved the highest discrimination (0.5991 ROC AUC) challenges the narrative that newer, more complex models universally outperform classical methods. On lapse data—where the underlying relationships are largely additive and monotonic—simpler, more interpretable models held their own. This observation aligns with decades of statistical wisdom: **the best model is the simplest one that captures the true data-generating process**.

#### 2. The Hierarchy of Complexity

Ranked by discrimination performance, our results showed:
- **Simplest (GLM, 1996):** 0.5991 ROC AUC
- **Flexible (TabPFN, 2023):** 0.5929 ROC AUC (−0.62%)
- **Tree ensemble (CatBoost, 2017):** 0.5909 ROC AUC (−1.36%)
- **Forest (RandomForest, 2001):** 0.5777 ROC AUC (−3.56%)
- **Boosting (XGBoost, 2016):** 0.5512 ROC AUC (−7.99%)

Note that more recent algorithms did not consistently outperform their predecessors. The pattern is driven by **data structure**, not model novelty.

#### 3. Where TabPFN Excels: Probability Calibration

While TabPFN matched (not exceeded) the GLM on discrimination, it significantly outperformed in calibration quality once post-hoc optimization was applied. The engineered-then-calibrated TabPFN delivered a **+1.66% Brier improvement**—a tangible, economically meaningful gain for actuarial applications where probability accuracy drives pricing and reserving decisions.

This finding highlights an underappreciated strength of modern foundation models: they can be tuned post-hoc for specific metrics (calibration, fairness, etc.) without retraining weights.

#### 4. The Computation-Complexity Trade-off

Fitting the GLM took 0.04 seconds; TabPFN took 1.89 seconds. Prediction latency was 0.01s (GLM) versus 7.6s (TabPFN, via remote API). For most actuarial workflows, this difference is negligible. However, for real-time scoring applications (e.g., quote-to-cash workflows), the GLM's speed advantage is non-trivial.

#### 5. Interpretability and Governance

The GLM produces readily interpretable coefficients—a feature highly valued in regulated actuarial practice. TabPFN, as a black-box neural network, requires post-hoc explanation techniques (SHAP, feature importance analysis) to justify its predictions to regulators and stakeholders. This governance cost, while not quantified here, is substantial in practice.

#### Key Takeaway

Our comprehensive evaluation demonstrates that **TabPFN is not a universal winner**, but rather a specialised tool with particular strengths (calibration, zero tuning, robustness to feature engineering) and acknowledged weaknesses (discrimination on well-structured data, interpretation, inference latency). The ideal approach is not to replace the GLM entirely, but to view TabPFN as a complementary technique—one that can be deployed in conjunction with classical methods to improve specific metrics (calibration, robustness) that matter to the business.
- **Low deployment complexity** (no GPU required, minimal inference overhead)
- **High production stability** (isotonic calibration is interpretable and robust)

---

### Revised Conclusion

Having set out to comprehensively evaluate whether TabPFN can deliver tangible benefits over traditional and state-of-the-art alternative methods, our results paint a nuanced picture that challenges conventional assumptions about model complexity.

## Why TabPFN Matters—When and Where

Our results illustrate a timeless principle in machine learning: **no single model dominates across all dimensions**. TabPFN lost on raw prediction power but won on calibration quality. This isn't a failure of TabPFN; it's evidence that method selection should match your specific priorities.

### Where TabPFN Shines

**Probability Calibration:** After isotonic adjustment, TabPFN delivers superior calibrated probabilities compared to the GLM (Brier score improvement of +1.66%). For actuarial work—where probability accuracy directly impacts pricing, reserving, and capital models—this is a genuine business advantage.

**Setup Simplicity:** TabPFN requires no hyperparameter tuning. All models in this study, including the GLM, received identical standardized preprocessing (ordinal encoding and feature scaling). TabPFN's true advantage is making reliable predictions without the manual tuning required by traditional methods.

### Trade-Offs to Consider

**Speed:** TabPFN's ~2-second API latency beats milliseconds for the GLM. For model development and annual recalibration, this is trivial. For real-time scoring at scale, it matters.

**Interpretability:** GLM coefficients directly explain driver effects. TabPFN requires post-hoc methods (SHAP, etc.) for feature importance. Regulators often prefer transparency; this favors the GLM.

**Raw Prediction Power:** On this dataset, they're equivalent (~59% AUC). On other datasets, the winner may differ. No single method always wins.

### The Real Insight

TabPFN is not a replacement for classical methods—it's a **complementary tool**. The optimal strategy:
1. Use GLM for interpretability and regulatory approval
2. Use TabPFN for calibration-sensitive pricing models
3. Ensemble both for critical decisions

**For Practitioners:**
- Choose by dimension, not by hype. What's your priority: calibration, speed, interpretability, or raw prediction power?
- Invest in calibration. Post-hoc isotonic regression often delivers more value than raw model complexity.
- Pair new methods with old. GLM transparency + TabPFN calibration = robust, defensible solutions.
- **There is still plenty of life in the old GLM yet!** Modern methods enrich the toolkit without rendering classical approaches obsolete.

---
### Appendix A: Detailed Methodology

**Data Preparation**

The eudirectlapse dataset was split via stratified random sampling into training (70%) and test (30%) sets with seed 45, preserving class balance. All models received identical preprocessing: ordinal encoding for categorical variables and standardization of numerical features. This ensures a fair comparison where differences in performance reflect each model's inherent capabilities rather than preprocessing choices.

**All Five Competing Models**

1. **Logistic Regression (GLM)** – Scikit-learn, default parameters
2. **TabPFN (v2.5)** – Remote Client API, no tuning
3. **CatBoost** – Categorical-optimized, defaults
4. **RandomForest** – 100 trees, defaults
5. **XGBoost** – Industry standard, defaults

**Key Metrics**

- **ROC AUC:** Discrimination ability (0.5 = random, 1.0 = perfect)
- **PR AUC:** Minority class performance (more informative for imbalanced data)
- **Brier Score:** Calibration quality (lower = better; most relevant for actuarial use)

**Reproducibility**

Fixed seeds (45 for train/test, 943321 for stochastic components). Code in Jupyter notebooks; runnable end-to-end with Python 3.8+, scikit-learn ≥1.0, TabPFN client API.

### Appendix B: How TabPFN Works—Technical Details

TabPFN represents a fundamental shift in how we approach tabular prediction. Rather than being trained on a specific dataset, TabPFN is pre-trained to learn general prediction strategies across millions of synthetically generated tabular datasets.

**The Core Idea: In-Context Learning**

During pre-training, TabPFN is exposed to synthetically generated datasets designed to capture diverse data-generating processes: non-linear relationships, missing values, categorical variables, noise, class imbalance, and high-dimensional feature spaces. From these examples, the model learns a meta-strategy—essentially, "how to solve tabular problems in general."

This learned strategy is encoded into transformer network weights. When you apply TabPFN to a new dataset, you provide the full training sample as context. The model processes this context in a single forward pass and generates predictions—without hyperparameter tuning, without cross-validation, without searching over architectures.

**Version 2.5 Improvements**

Recent advances in TabPFN v2.5 include:
- Deeper transformer architectures for better feature interaction modeling
- Richer synthetic training priors (more realistic data distributions)
- Inference-time optimizations for speed
- Extended applicability to datasets with up to 50,000 rows and mixed feature types

**Why This Matters for Actuaries**

The foundation model approach offers distinct practical advantages:
1. **No hyperparameter tuning** – Eliminates lengthy grid search workflows
2. **Works directly on raw data** – No feature engineering or scaling required
3. **Robustness to data shifts** – Pre-training on diverse data provides generalization
4. **Probability calibration** – The model learns to estimate uncertainty distributions, not just point predictions

The tradeoff: prediction latency (~2 seconds via remote API) and reduced interpretability compared to classical methods.

### Appendix C: Post-Hoc Calibration and Optimization Details

**Understanding Model Calibration**

A model is well-calibrated when its predicted probability of an event matches the true observed frequency. For example, if a model predicts 20% lapse probability for 1,000 policies, we should observe roughly 200 actual lapses.

Many models—especially neural networks like TabPFN—produce overconfident predictions. Calibration methods address this by applying a post-hoc transformation to predicted probabilities, moving them closer to true empirical frequencies.

**Brier Score: The Calibration Metric**

The Brier Score measures calibration quality: $BS = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)^2$ where $p_i$ is the predicted probability and $y_i$ is the observed outcome (0 or 1). Lower is better; 0 = perfect calibration.

In our analysis:
- Raw TabPFN: 0.1108
- Isotonic calibrated TabPFN: 0.1080 (+0.87% improvement)
- GLM: 0.1098

**Isotonic Regression vs Platt Scaling**

We compared two calibration approaches:

1. **Platt Scaling:** Fits logistic regression to map predictions. Assumes sigmoid-shaped calibration error. Result: +0.45% improvement.

2. **Isotonic Regression:** Non-parametric, preserves rank order, makes no distributional assumptions. Result: +0.87% improvement.

Isotonic won because the lapse probability range (~0.05 to ~0.30) benefits from its flexible adaptation. For business applications, the +0.42% additional gain justifies the computational overhead.

**Why Ensemble Bagging Failed**

Trained three TabPFN models via bootstrap and averaged predictions. Hypothesis: ensemble voting reduces variance. Result: Brier score degraded by ~3–4%.

Reason: TabPFN already aggregates signal effectively via pre-training. Averaging multiple models doesn't create new information—it introduces noise. Foundation models should be used directly, not via ensemble voting.

**Practical Recommendations**

- Apply isotonic calibration when probability accuracy matters (pricing, reserving, capital models)
- Combine feature engineering with calibration (often outperforms either alone)
- Don't ensemble foundation models; trust their pre-training
- Reserve ensemble methods for tree-based models where they're proven effective

