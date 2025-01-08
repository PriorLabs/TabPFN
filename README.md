# TabPFN

[![PyPI version](https://badge.fury.io/py/tabpfn.svg)](https://badge.fury.io/py/tabpfn)
[![Downloads](https://pepy.tech/badge/tabpfn)](https://pepy.tech/project/tabpfn)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.com/channels/1285598202732482621/)
[![Documentation](https://img.shields.io/badge/docs-priorlabs.ai-blue)](https://priorlabs.ai/)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/194mCs6SEPEW6C0rcP7xWzcEtt1RBc8jJ)

TabPFN is a foundation model for tabular data that outperforms traditional methods while 
being dramatically faster. This repository contains the core PyTorch implementation with
CUDA optimization.

⚠️ **Major Update: Version 2.0**

Complete codebase overhaul with new architecture and features. Previous version available at [v1.0.0](../../tree/v1.0.0).

## 🌐 TabPFN Ecosystem

Choose the right TabPFN implementation for your needs:

- **[TabPFN Client](https://github.com/automl/tabpfn-client)**: Easy-to-use API client for cloud-based inference
- **[TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions)**: Community extensions and integrations
- **TabPFN (this repo)**: Core implementation for local deployment and research

Try our [Interactive Colab Tutorial](https://colab.research.google.com/drive/194mCs6SEPEW6C0rcP7xWzcEtt1RBc8jJ) to get started quickly.

## 🏁 Quick Start

### Installation

```bash
# Simple installation
pip install tabpfn

# Development installation
pip install -e ".[dev]"
pre-commit install
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
- **Automatic ensembling**: Controls with `N_ensemble_configurations`
- **Independent predictions**: Test samples can be predicted individually or in batch
- **Differentiable**: Core model is differentiable (except preprocessing)
- **GPU Support**: Use `device='cuda'` for GPU acceleration

## 📜 License

TBD, currently proprietary software all rights reserved.
Copyright (c) 2025 Prior Labs GmbH


## 📚 Citation

TBD


## 🤝 Join Our Community

We're building the future of tabular machine learning and would love your involvement:

1. **Connect & Learn**: 
   - Join our [Discord Community](https://discord.gg/VJRuU3bSxt)
   - Read our [Documentation](https://priorlabs.ai/)
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

Contribution guidelines:
- Follow PEP 8 style guide (enforced by pre-commit hooks)
- Add tests for new functionality
- Update documentation as needed
- Sign CLA before submitting PRs
- Open issues for feature discussions

---

Built with ❤️ by [Prior Labs](https://priorlabs.ai)
