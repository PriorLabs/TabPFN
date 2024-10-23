# TabPFN

The TabPFN is a neural network that learned to do tabular data prediction.
This is the original CUDA-supporting pytorch impelementation.

We created a [Colab](https://colab.research.google.com/drive/194mCs6SEPEW6C0rcP7xWzcEtt1RBc8jJ), that lets you play with our scikit-learn interface.

**Join our discord here: https://discord.gg/VJRuU3bSxt**

## Installation

```bash
pip install tabpfn
```

If you want to train and evaluate our method like we did in the paper (including baselines) please install with
```bash
pip install tabpfn[full]
```
To run the autogluon and autosklearn baseline please create a separate environment and install autosklearn==0.14.5 / autogluon==0.4.0, installation in the same environment as our other baselines is not possible.

## Getting started

A simple usage of our sklearn interface is:
```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
# When N_ensemble_configurations > #features * #classes, no further averaging is applied.

classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))
```

### TabPFN Usage

TabPFN is different from other methods you might know for tabular classification.
Here, we list some tips and tricks that might help you understand how to use it best.

- Do not preprocess inputs to TabPFN. TabPFN pre-processes inputs internally. It applies a z-score normalization (`x-train_x.mean()/train_x.std()`) per feature (fitted on the training set) and log-scales outliers [heuristically](https://github.com/automl/TabPFN/blob/f7402ec1916aa78d953574daf95508045af5953e/tabpfn/utils.py#L201). Finally, TabPFN  applies a PowerTransform to all features for every second ensemble member. Pre-processing is important for the TabPFN to make sure that the real-world dataset lies in the distribution of the synthetic datasets seen during training. So to get the best results, do not apply a PowerTransformation to the inputs.
- TabPFN expects scalar values only (if your categoricals are `float`s just leave them as they are, if you have categoricals that are not encoded as `float` (rather `str` or `object`), encode your categoricals e.g. with [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder)). TabPFN works best on data that does not contain any categorical or NaN data (see [Appendix B.1](https://arxiv.org/abs/2207.01848)).
- TabPFN ensembles multiple input encodings per default. It feeds different index rotations of the features and labels to the model per ensemble member. You can control the ensembling with `TabPFNClassifier(...,N_ensemble_configurations=?)`
- TabPFN does not use any statistics from the test set. That means predicting each test example one-by-one will yield the same result as feeding the whole test set together.
- TabPFN is differentiable in principle, only the pre-processing is not and relies on numpy.

## Our Paper
Read our [paper](https://arxiv.org/abs/2207.01848) for more information about the setup (or contact us ☺️).
If you use our method, please cite us using
```
@inproceedings{
  hollmann2023tabpfn,
  title={Tab{PFN}: A Transformer That Solves Small Tabular Classification Problems in a Second},
  author={Noah Hollmann and Samuel M{\"u}ller and Katharina Eggensperger and Frank Hutter},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=cp5PvcI6w8_}
}
```

## License
Copyright 2022 Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
