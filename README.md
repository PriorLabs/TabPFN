# TabPFN

[![PyPI version](https://badge.fury.io/py/tabpfn.svg)](https://badge.fury.io/py/tabpfn)
[![Downloads](https://pepy.tech/badge/tabpfn)](https://pepy.tech/project/tabpfn)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.com/channels/1285598202732482621/)
[![Documentation](https://img.shields.io/badge/docs-priorlabs.ai-blue)](https://priorlabs.ai/docs)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://tinyurl.com/tabpfn-colab-local)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/tabpfn/)

<img src="https://github.com/PriorLabs/tabpfn-extensions/blob/main/tabpfn_summary.webp" width="80%" alt="TabPFN Summary">

⚠️ **Major Update: Version 2.0:** Complete codebase overhaul with new architecture and 
features. Previous version available at [v1.0.0](../../tree/v1.0.0) and 
`pip install tabpfn==0.1.11`.

📚 For detailed usage examples and best practices, check out [Interactive Colab Tutorial](https://tinyurl.com/tabpfn-colab-local)

## 🏁 Quick Start

TabPFN is a foundation model for tabular data that outperforms traditional methods while 
being dramatically faster. This repository contains the core PyTorch implementation with
CUDA optimization.

> ⚡ **GPU Recommended**:  
> For optimal performance, use a GPU (even older ones with ~8GB VRAM work well; 16GB needed for some large datasets).  
> On CPU, only small datasets (≲1000 samples) are feasible.  
> No GPU? Use our free hosted inference via [TabPFN Client](https://github.com/PriorLabs/tabpfn-client).

### Installation
Official installation (pip)
```bash
pip install tabpfn
```
OR installation from source
```bash
pip install "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git"
```
OR local development installation
```bash

git clone https://github.com/PriorLabs/TabPFN.git
pip install -e "TabPFN[dev]"
```

### Basic Usage

#### Classification
```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize a classifier
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

#### Regression
```python
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Assuming there is a TabPFNRegressor (if not, a different regressor should be used)
from tabpfn import TabPFNRegressor  

# Load Boston Housing data
df = fetch_openml(data_id=531, as_frame=True)  # Boston Housing dataset
X = df.data
y = df.target.astype(float)  # Ensure target is float for regression

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize the regressor
regressor = TabPFNRegressor()  
regressor.fit(X_train, y_train)

# Predict on the test set
predictions = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)
```

### Best Results

For optimal performance, use the `AutoTabPFNClassifier` or `AutoTabPFNRegressor` for post-hoc ensembling. These can be found in the [TabPFN Extensions](https://github.com/PriorLabs/tabpfn-extensions) repository. Post-hoc ensembling combines multiple TabPFN models into an ensemble. 

**Steps for Best Results:**
1. Install the extensions:
   ```bash
   git clone https://github.com/priorlabs/tabpfn-extensions.git
   pip install -e tabpfn-extensions
   ```

2.
   ```python 
   from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

   clf = AutoTabPFNClassifier(max_time=120, device="cuda") # 120 seconds tuning time
   clf.fit(X_train, y_train)
   predictions = clf.predict(X_test)
   ```

## 🌐 TabPFN Ecosystem

Choose the right TabPFN implementation for your needs:

- **[TabPFN Client](https://github.com/priorlabs/tabpfn-client)**  
  Simple API client for using TabPFN via cloud-based inference.

- **[TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions)**  
  A powerful companion repository packed with advanced utilities, integrations, and features - great place to contribute:

  - 🔍 **`interpretability`**: Gain insights with SHAP-based explanations, feature importance, and selection tools.
  - 🕵️‍♂️ **`unsupervised`**: Tools for outlier detection and synthetic tabular data generation.
  - 🧬 **`embeddings`**: Extract and use TabPFN’s internal learned embeddings for downstream tasks or analysis.
  - 🧠 **`many_class`**: Handle multi-class classification problems that exceed TabPFN's built-in class limit.
  - 🌲 **`rf_pfn`**: Combine TabPFN with traditional models like Random Forests for hybrid approaches.
  - ⚙️ **`hpo`**: Automated hyperparameter optimization tailored to TabPFN.
  - 🔁 **`post_hoc_ensembles`**: Boost performance by ensembling multiple TabPFN models post-training.

  ✨ To install:
  ```bash
  git clone https://github.com/priorlabs/tabpfn-extensions.git
  pip install -e tabpfn-extensions
  ```

- **[TabPFN (this repo)](https://github.com/priorlabs/tabpfn)**  
  Core implementation for fast and local inference with PyTorch and CUDA support.

- **[TabPFN UX](https://ux.priorlabs.ai)**  
  No-code graphical interface to explore TabPFN capabilities—ideal for business users and prototyping.

## 📜 License

Prior Labs License (Apache 2.0 with additional attribution requirement): [here](https://priorlabs.ai/tabpfn-license/)

## 🤝 Join Our Community

We're building the future of tabular machine learning and would love your involvement:

1. **Connect & Learn**: 
   - Join our [Discord Community](https://discord.gg/VJRuU3bSxt)
   - Read our [Documentation](https://priorlabs.ai/docs)
   - Check out [GitHub Issues](https://github.com/priorlabs/tabpfn/issues)

2. **Contribute**: 
   - Report bugs or request features
   - Submit pull requests
   - Share your research and use cases

3. **Stay Updated**: Star the repo and join Discord for the latest updates

## 📚 Citation

You can read our paper explaining TabPFN [here](https://doi.org/10.1038/s41586-024-08328-6). 

```bibtex
@article{hollmann2025tabpfn,
 title={Accurate predictions on small data with a tabular foundation model},
 author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and
         Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and
         Schirrmeister, Robin Tibor and Hutter, Frank},
 journal={Nature},
 year={2025},
 month={01},
 day={09},
 doi={10.1038/s41586-024-08328-6},
 publisher={Springer Nature},
 url={https://www.nature.com/articles/s41586-024-08328-6},
}

@inproceedings{hollmann2023tabpfn,
  title={TabPFN: A transformer that solves small tabular classification problems in a second},
  author={Hollmann, Noah and M{\"u}ller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  booktitle={International Conference on Learning Representations 2023},
  year={2023}
}
```



## ❓ FAQ

### **Usage & Compatibility**

**Q: What dataset sizes work best with TabPFN?**  
A: TabPFN is optimized for **datasets up to 10,000 rows**. For larger datasets, consider using **Random Forest preprocessing** or other extensions. See our [Colab notebook](https://colab.research.google.com/drive/154SoIzNW1LHBWyrxNwmBqtFAr1uZRZ6a#scrollTo=OwaXfEIWlhC8) for strategies.

**Q: Why can't I use TabPFN with Python 3.8?**  
A: TabPFN v2 requires **Python 3.9+** due to newer language features. Compatible versions: **3.9, 3.10, 3.11, 3.12, 3.13**.

### **Installation & Setup**

**Q: How do I use TabPFN without an internet connection?**  

TabPFN automatically downloads model weights when first used. For offline usage:

**Using the Provided Download Script**

If you have the TabPFN repository, you can use the included script to download all models (including ensemble variants):

```bash
# After installing TabPFN
python scripts/download_all_models.py
```

This script will download the main classifier and regressor models, as well as all ensemble variant models to your system's default cache directory.

**Manual Download**

1. Download the model files manually from HuggingFace:
   - Classifier: [tabpfn-v2-classifier.ckpt](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/main/tabpfn-v2-classifier.ckpt)
   - Regressor: [tabpfn-v2-regressor.ckpt](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/resolve/main/tabpfn-v2-regressor.ckpt)

2. Place the file in one of these locations:
   - Specify directly: `TabPFNClassifier(model_path="/path/to/model.ckpt")`
   - Set environment variable: `os.environ["TABPFN_MODEL_CACHE_DIR"] = "/path/to/dir"`
   - Default OS cache directory:
     - Windows: `%APPDATA%\tabpfn\`
     - macOS: `~/Library/Caches/tabpfn/`
     - Linux: `~/.cache/tabpfn/`

**Q: I'm getting a `pickle` error when loading the model. What should I do?**  
A: Try the following:
- Download the newest version of tabpfn `pip install tabpfn --upgrade`
- Ensure model files downloaded correctly (re-download if needed)

### **Performance & Limitations**

**Q: Can TabPFN handle missing values?**  
A: **Yes!**

**Q: How can I improve TabPFN’s performance?**  
A: Best practices:
- Use **AutoTabPFNClassifier** from [TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions) for post-hoc ensembling
- Feature engineering: Add domain-specific features to improve model performance  
Not effective:
  - Adapt feature scaling
  - Convert categorical features to numerical values (e.g., one-hot encoding)

## 🛠️ Development

1. Setup environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
git clone https://github.com/PriorLabs/TabPFN.git
cd tabpfn
pip install -e ".[dev]"
pre-commit install
```

2. Before committing:
```bash
pre-commit run --all-files
```

3. Run tests:
```bash
pytest tests/
```

---

Built with ❤️ by [Prior Labs](https://priorlabs.ai) - Copyright (c) 2025 Prior Labs GmbH
