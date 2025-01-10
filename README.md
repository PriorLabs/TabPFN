# TabPFN

[![PyPI version](https://badge.fury.io/py/tabpfn.svg)](https://badge.fury.io/py/tabpfn)
[![Downloads](https://pepy.tech/badge/tabpfn)](https://pepy.tech/project/tabpfn)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.com/channels/1285598202732482621/)
[![Documentation](https://img.shields.io/badge/docs-priorlabs.ai-blue)](https://priorlabs.ai/)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://tinyurl.com/tabpfn-colab-local)

TabPFN is a foundation model for tabular data that outperforms traditional methods while 
being dramatically faster. This repository contains the core PyTorch implementation with
CUDA optimization.

⚠️ **Major Update: Version 2.0:** Complete codebase overhaul with new architecture and 
features. Previous version available at [v1.0.0](../../tree/v1.0.0) and 
`pip install tabpfn<2`.

📚 For detailed usage examples and best practices, check out:
- [Interactive Colab Tutorial](https://tinyurl.com/tabpfn-colab-local)

## 🌐 TabPFN Ecosystem

Choose the right TabPFN implementation for your needs:

- **[TabPFN Client](https://github.com/automl/tabpfn-client)**: Easy-to-use API client for cloud-based inference
- **[TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions)**: Community extensions and integrations
- **TabPFN (this repo)**: Core implementation for local deployment and research

Try our [Interactive Colab Tutorial](https://colab.research.google.com/drive/1SHa43VuHASLjevzO7y3-wPCxHY18-2H6?usp=sharing) to get started quickly.

## 🏁 Quick Start

### Installation

```bash
# Simple installation
pip install tabpfn

# Local development installation
git clone https://github.com/PriorLabs/TabPFN.git
pip install -e "tabpfn[dev]"
```

### Basic Usage

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

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

## 💡 Usage Tips

TabPFN is designed to work out-of-the-box with minimal preprocessing:

- **No preprocessing needed**: TabPFN handles normalization internally
- **Categorical variables**: Use numerical encodings (floats for ordered, OrdinalEncoder for unordered)
- **Automatic ensembling**: Controls with `n_estimators`
- **Independent predictions**: Test samples can be predicted individually or in batch
- **Differentiable**: Core model is differentiable (except preprocessing)
- **GPU Support**: Use `device='cuda'` for GPU acceleration

## 📜 License

[Prior Labs License (Apache 2.0 with additional attribution requirement)](https://priorlabs.ai/tabpfn-license/)

## 📚 Citation

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
```

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
