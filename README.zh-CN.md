# TabPFN

[![PyPI 版本](https://badge.fury.io/py/tabpfn.svg)](https://badge.fury.io/py/tabpfn)
[![下载量](https://pepy.tech/badge/tabpfn)](https://pepy.tech/project/tabpfn)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.gg/BHnX2Ptf4j)
[![文档](https://img.shields.io/badge/docs-priorlabs.ai-blue)](https://priorlabs.ai/docs)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb)
[![Python 版本](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/tabpfn/)

<img src="https://github.com/PriorLabs/tabpfn-extensions/blob/main/tabpfn_summary.webp" width="80%" alt="TabPFN Summary">

## 快速开始

### 交互式笔记本教程
> [!TIP]
>
> 通过我们的交互式 Colab 笔记本直接上手！这是获得 TabPFN 实践体验的最佳方式，将引导您完成安装、分类和回归示例。
>
> [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb)

> ⚡ **推荐使用 GPU**：
> 为了获得最佳性能，建议使用 GPU（即使是较旧的约 8GB 显存的 GPU也能很好地工作；某些大型数据集需要 16GB）。
> 在 CPU 上，仅小数据集（≲1000 样本）可行。
> 没有 GPU？请通过 [TabPFN Client](https://github.com/PriorLabs/tabpfn-client) 使用我们免费的托管推理服务。

### 安装
官方安装（pip）
```bash
pip install tabpfn
```
或者从源码安装
```bash
pip install "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git"
```
或者本地开发安装：首先[安装 uv](https://docs.astral.sh/uv/getting-started/installation)（推荐使用 0.10.0 或更高版本），然后运行
```bash
git clone https://github.com/PriorLabs/TabPFN.git --depth 1
cd TabPFN
uv sync
```

### 基本用法

使用默认的 TabPFN-2.6 模型（完全基于合成数据训练）：

```python
from tabpfn import TabPFNClassifier, TabPFNRegressor

clf = TabPFNClassifier()
clf.fit(X_train, y_train)  # 首次使用时下载检查点
predictions = clf.predict(X_test)

reg = TabPFNRegressor()
reg.fit(X_train, y_train)  # 首次使用时下载检查点
predictions = reg.predict(X_test)
```

使用其他模型版本（例如 TabPFN-2.5）：

```python
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

classifier = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)
regressor = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5)
```

完整的示例请参见 [tabpfn_for_binary_classification.py](https://github.com/PriorLabs/TabPFN/tree/main/examples/tabpfn_for_binary_classification.py)、[tabpfn_for_multiclass_classification.py](https://github.com/PriorLabs/TabPFN/tree/main/examples/tabpfn_for_multiclass_classification.py) 和 [tabpfn_for_regression.py](https://github.com/PriorLabs/TabPFN/tree/main/examples/tabpfn_for_regression.py) 文件。


### 使用技巧

- **使用批量预测模式**：每次 `predict` 调用都会重新计算训练集。分别对 100 个样本调用 100 次 `predict` 比单次调用慢且贵近 100 倍。如果测试集非常大，请将其分成每批 1000 个样本的块。
- **避免数据预处理**：向模型输入数据时不要进行数据缩放或独热编码。
- **使用 GPU**：TabPFN 在 CPU 上执行速度很慢。请确保有 GPU 可用以获得更好的性能。
- **注意数据集大小**：TabPFN 在少于 100,000 个样本和 2000 个特征的数据集上效果最佳。对于更大的数据集，我们建议查看[大数据集指南](https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/large_datasets/large_datasets_example.py)。

## TabPFN 生态系统

根据您的需求选择合适的 TabPFN 实现：

- **[TabPFN Client](https://github.com/priorlabs/tabpfn-client)**
  简单的 API 客户端，通过基于云的推理服务使用 TabPFN。

- **[TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions)**
  一个强大的配套仓库，包含高级工具、集成和功能——非常适合贡献代码：

  -  **`interpretability`**：通过基于 SHAP 的解释、特征重要性和选择工具获取洞察。
  -  **`unsupervised`**：用于异常值检测和合成表格数据生成的工具。
  -  **`embeddings`**：提取并使用 TabPFN 内部学习的嵌入，用于下游任务或分析。
  -  **`many_class`**：处理超过 TabPFN 内置类别限制的多分类问题。
  -  **`rf_pfn`**：将 TabPFN 与随机森林等传统模型结合，用于混合方法。
  -  **`hpo`**：专为 TabPFN 定制的自动超参数优化。
  -  **`post_hoc_ensembles`**：通过训练后集成多个 TabPFN 模型来提升性能。

  安装方式：
  ```bash
  git clone https://github.com/priorlabs/tabpfn-extensions.git
  pip install -e tabpfn-extensions
  ```

- **[TabPFN（本仓库）](https://github.com/priorlabs/tabpfn)**
  核心实现，支持通过 PyTorch 和 CUDA 进行快速本地推理。

- **[TabPFN UX](https://ux.priorlabs.ai)**
  无代码图形界面，用于探索 TabPFN 功能——非常适合业务用户和原型开发。

## TabPFN 工作流程一览
按照此决策树构建模型并从我们的生态系统中选择合适的扩展。它将引导您回答关于数据、硬件和性能需求的关键问题，指导您找到最适合特定用例的解决方案。

```mermaid
---\nconfig:\n  theme: 'default'\n  themeVariables:\n    edgeLabelBackground: 'white'\n---\ngraph LR\n    %% 1. 定义配色方案和样式\n    classDef default fill:#fff,stroke:#333,stroke-width:2px,color:#333;\n    classDef start_node fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#333;\n    classDef process_node fill:#e0f2f1,stroke:#00796b,stroke-width:2px,color:#333;\n    classDef decision_node fill:#fff8e1,stroke:#ffa000,stroke-width:2px,color:#333;\n\n    style Infrastructure fill:#fff,stroke:#ccc,stroke-width:5px;\n    style Unsupervised fill:#fff,stroke:#ccc,stroke-width:5px;\n    style Data fill:#fff,stroke:#ccc,stroke-width:5px;\n    style Performance fill:#fff,stroke:#ccc,stroke-width:5px;\n    style Interpretability fill:#fff,stroke:#ccc,stroke-width:5px;\n\n    %% 2. 定义图结构\n    subgraph Infrastructure [基础设施]\n        start((开始)) --> gpu_check["GPU 是否可用？"];\n        gpu_check -- 是 --> local_version["使用 TabPFN<br/>(本地 PyTorch)"];\n        gpu_check -- 否 --> api_client["使用 TabPFN-Client<br/>(云端 API)"];\n        task_type["您的任务是？"]
    end\n\n    local_version --> task_type\n    api_client --> task_type\n\n    end_node((工作流<br/>完成));\n\n    subgraph Unsupervised [无监督]\n        unsupervised_type["选择<br/>无监督任务"];\n        unsupervised_type --> imputation["缺失值填充"]
        unsupervised_type --> data_gen["数据<br/>生成"];\n        unsupervised_type --> tabebm["数据<br/>增强"];\n        unsupervised_type --> density["异常值<br/>检测"];\n        unsupervised_type --> embedding["获取<br/>嵌入"];\n    end\n\n\n    subgraph Data [数据]\n        data_check["数据检查"];\n        model_choice["样本数 > 50k 或<br/>类别数 > 10？"];\n        data_check -- "表格包含文本数据？" --> api_backend_note["注意：API 客户端具有<br/>原生文本支持"];\n        api_backend_note --> model_choice;\n        data_check -- "时间序列数据？" --> ts_features["使用时间序列<br/>特征"];\n        ts_features --> model_choice;\n        data_check -- "纯表格数据" --> model_choice;\n        model_choice -- "否" --> finetune_check;\n        model_choice -- "是，50k-100k 样本" --> ignore_limits["设置<br/>ignore_pretraining_limits=True"];\n        model_choice -- "是，>100k 样本" --> subsample["大数据集指南<br/>"];\n        model_choice -- "是，>10 类别" --> many_class["多类别<br/>方法"];\n    end\n\n    subgraph Performance [性能]\n        finetune_check["需要<br/>微调？"];\n        performance_check["需要更好的性能？"];\n        speed_check["预测时需要<br/>更快的推理速度？"];\n        kv_cache["启用 KV 缓存<br/>(fit_mode='fit_with_cache')<br/><small>预测更快；内存占用 ~O(N×F)</small>"];\n        tuning_complete["调优完成"];\n\n        finetune_check -- 是 --> finetuning["微调"];\n        finetune_check -- 否 --> performance_check;\n\n        finetuning --> performance_check;\n\n        performance_check -- 否 --> tuning_complete;\n        performance_check -- 是 --> hpo["超参数优化 (HPO)"];\n        performance_check -- 是 --> post_hoc["训练后<br/>集成"];\n        performance_check -- 是 --> more_estimators["更多<br/>估计器"];\n        performance_check -- 是 --> speed_check;\n\n        speed_check -- 是 --> kv_cache;\n        speed_check -- 否 --> tuning_complete;\n\n        hpo --> tuning_complete;\n        post_hoc --> tuning_complete;\n        more_estimators --> tuning_complete;\n        kv_cache --> tuning_complete;\n    end\n\n    subgraph Interpretability [可解释性]\n\n        tuning_complete --> interpretability_check;\n\n        interpretability_check["需要<br/>可解释性？"];\n\n        interpretability_check --> feature_selection["特征选择"];\n        interpretability_check --> partial_dependence["部分依赖图"];\n        interpretability_check --> shapley["使用 SHAP<br/>解释"];\n        interpretability_check --> shap_iq["使用 SHAP IQ<br/>解释"];\n        interpretability_check -- 否 --> end_node;\n\n        feature_selection --> end_node;\n        partial_dependence --> end_node;\n        shapley --> end_node;\n        shap_iq --> end_node;\n\n    end\n\n    %% 3. 链接子图和路径\n    task_type -- "分类或回归" --> data_check;\n    task_type -- "无监督" --> unsupervised_type;\n\n    subsample --> finetune_check;\n    ignore_limits --> finetune_check;\n    many_class --> finetune_check;\n\n    %% 4. 应用样式\n    class start,end_node start_node;\n    class local_version,api_client,imputation,data_gen,tabebm,density,embedding,api_backend_note,ts_features,subsample,ignore_limits,many_class,finetuning,feature_selection,partial_dependence,shapley,shap_iq,hpo,post_hoc,more_estimators,kv_cache process_node;\n    class gpu_check,task_type,unsupervised_type,data_check,model_choice,finetune_check,interpretability_check,performance_check,speed_check decision_node;\n    class tuning_complete process_node;\n\n    %% 5. 添加可点击链接\n    click local_version "https://github.com/PriorLabs/TabPFN" "TabPFN 后端选项"\n    click api_client "https://github.com/PriorLabs/tabpfn-client" "TabPFN API 客户端"\n    click api_backend_note "https://github.com/PriorLabs/tabpfn-client" "TabPFN API 后端"\n    click unsupervised_type "https://github.com/PriorLabs/tabpfn-extensions" "TabPFN 扩展"\n    click imputation "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/imputation.py" "TabPFN 缺失值填充示例"\n    click data_gen "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/generate_data.py" "TabPFN 数据生成示例"\n    click tabebm "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/tabebm/tabebm_augment_real_world_data.ipynb" "TabEBM 数据增强示例"\n    click density "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/unsupervised/density_estimation_outlier_detection.py" "TabPFN 密度估计/异常检测示例"\n    click embedding "https://github.com/PriorLabs/tabpfn-extensions/tree/main/examples/embedding" "TabPFN 嵌入示例"\n    click ts_features "https://github.com/PriorLabs/tabpfn-time-series" "TabPFN 时间序列示例"\n    click many_class "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/many_class/many_class_classifier_example.py" "多类别示例"\n    click finetuning "https://github.com/PriorLabs/TabPFN/blob/main/examples/finetune_classifier.py" "微调示例"\n    click feature_selection "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/feature_selection.py" "特征选择示例"\n    click partial_dependence "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/pdp_example.py" "部分依赖图示例"\n    click shapley "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/shap_example.py" "Shapley 值示例"\n    click shap_iq "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/interpretability/shapiq_example.py" "SHAP IQ 示例"\n    click post_hoc "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/phe/phe_example.py" "训练后集成示例"\n    click hpo "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/hpo/tuned_tabpfn.py" "HPO 示例"\n    click subsample "https://github.com/PriorLabs/tabpfn-extensions/blob/main/examples/large_datasets/large_datasets_example.py" "大数据集示例"\n    click kv_cache "https://github.com/PriorLabs/TabPFN/blob/main/examples/kv_cache_fast_prediction.py" "KV 缓存快速预测示例"

```

## 许可证

TabPFN-2.5 和 TabPFN-2.6 模型权重采用[非商业许可证](https://huggingface.co/Prior-Labs/tabpfn_2_6/blob/main/LICENSE)。这些是默认使用的。

代码和 TabPFN-2 模型权重采用 Prior Labs 许可证（Apache 2.0 附加归属要求）：[此处](LICENSE)。要使用 v2 模型权重，请按如下方式实例化模型：

```
from tabpfn.constants import ModelVersion

tabpfn_v2 = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
```

## 企业与生产环境

对于高吞吐量或大规模生产环境，我们提供**企业版**，具有以下能力：
-   **快速推理模式**：一种专有的蒸馏引擎，可将 TabPFN-2.6 转换为紧凑的 MLP 或树集成，为实时应用提供数量级更低的延迟。
-   **大数据模式（扩展模式）**：一种高级操作模式，取消行数限制，支持高达**1000 万行**的数据集——比默认的 TabPFN-2.5 和 TabPFN-2.6 模型增加 1000 倍。
-   **商业支持**：包含用于生产用例的商业企业许可证、专门的集成支持以及访问私有高速推理引擎。

**了解更多或申请商业许可证，请通过 [sales@priorlabs.ai](mailto:sales@priorlabs.ai) 联系我们。**


## 加入我们的社区

我们正在构建表格机器学习的未来，非常希望您的参与：

1. **联系与学习**：
   - 加入我们的 [Discord 社区](https://discord.gg/VJRuU3bSxt)
   - 阅读我们的[文档](https://priorlabs.ai/docs)
   - 查看 [GitHub Issues](https://github.com/priorlabs/tabpfn/issues)

2. **贡献**：
   - 报告错误或请求功能
   - 提交 pull 请求（请确保先打开 issue 讨论该功能/错误（如果不存在的话））
   - 分享您的研究和用例

3. **保持更新**：给仓库加星标并加入 Discord 以获取最新更新

## 引用

您可以在[此处](https://doi.org/10.1038/s41586-024-08328-6)阅读解释 TabPFNv2 的论文，TabPFN-2.5 的模型报告可在[此处](https://arxiv.org/abs/2511.08667)获取。

```bibtex
@misc{grinsztajn2025tabpfn,
  title={TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models},
  author={Léo Grinsztajn and Klemens Flöge and Oscar Key and Felix Birkel and Philipp Jund and Brendan Roof and
          Benjamin Jäger and Dominik Safaric and Simone Alessi and Adrian Hayler and Mihir Manium and Rosen Yu and
          Felix Jablonski and Shi Bin Hoo and Anurag Garg and Jake Robertson and Magnus Bühler and Vladyslav Moroshan and
          Lennart Purucker and Clara Cornu and Lilly Charlotte Wehrhahn and Alessandro Bonetto and
          Bernhard Schölkopf and Sauraj Gambhir and Noah Hollmann and Frank Hutter},
  year={2025},
  eprint={2511.08667},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2511.08667},
}

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



## ❓ 常见问题

### **使用与兼容性**

**问：什么规模的数据集最适合 TabPFN？**
答：TabPFN-2.5 针对**最多 50,000 行**的数据集进行了优化。对于更大的数据集，请考虑使用**随机森林预处理**或其他扩展。请参阅我们的 [Colab 笔记本](https://colab.research.google.com/drive/154SoIzNW1LHBWyrxNwmBqtFAr1uZRZ6a#scrollTo=OwaXfEIWlhC8)了解策略。

**问：为什么我不能将 TabPFN 与 Python 3.8 一起使用？**
答：TabPFN 需要 **Python 3.9+**，因为使用了较新的语言特性。兼容版本：**3.9、3.10、3.11、3.12、3.13**。

### **安装与设置**

**问：如何获取 TabPFN-2.5 / TabPFN-2.6 的访问权限？**

首次使用时，TabPFN 会自动打开一个浏览器窗口，您可以通过 [PriorLabs](https://ux.priorlabs.ai) 登录并接受许可条款。您的身份验证令牌会缓存在本地，因此只需执行一次。

**对于无头 / CI 环境**（没有浏览器可用），请访问 [https://ux.priorlabs.ai](https://ux.priorlabs.ai)，进入**许可证**标签接受许可，然后使用从您的账户获取的令牌设置 `TABPFN_TOKEN` 环境变量。

如果无法使用基于浏览器的流程，请通过 [`sales@priorlabs.ai`](mailto:sales@priorlabs.ai) 联系我们。

**问：如何在没有互联网连接的情况下使用 TabPFN？**

TabPFN 在首次使用时自动下载模型权重。要离线使用：

**使用提供的下载脚本**

如果您有 TabPFN 仓库，可以使用附带的脚本下载所有模型（包括集成变体）：

```bash
# 安装 TabPFN 后
python scripts/download_all_models.py
```

此脚本将把主分类器和回归器模型以及所有集成变体模型下载到系统的默认缓存目录。

**手动下载**

1. 从 HuggingFace 手动下载模型文件：
   - 分类器：[tabpfn-v2.5-classifier-v2.5_default.ckpt](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/tabpfn-v2.5-classifier-v2.5_default.ckpt)（注意：默认分类器使用在真实数据上微调的模型）。
   - 回归器：[tabpfn-v2.5-regressor-v2.5_default.ckpt](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/tabpfn-v2.5-regressor-v2.5_default.ckpt)

2. 将文件放置在以下位置之一：
   - 直接指定：`TabPFNClassifier(model_path="/path/to/model.ckpt")`
   - 设置环境变量：`export TABPFN_MODEL_CACHE_DIR="/path/to/dir"`（见下面的环境变量 FAQ）
   - 默认操作系统缓存目录：
     - Windows: `%APPDATA%\tabpfn\`
     - macOS: `~/Library/Caches/tabpfn/`
     - Linux: `~/.cache/tabpfn/`

**问：加载模型时出现 `pickle` 错误。我该怎么办？**
答：请尝试以下操作：
- 下载最新版本的 tabpfn `pip install tabpfn --upgrade`
- 确保模型文件正确下载（必要时重新下载）

**问：有哪些环境变量可用于配置 TabPFN？**
答：TabPFN 使用 Pydantic settings 进行配置，支持环境变量和 `.env` 文件：

**身份验证：**
- `TABPFN_TOKEN`：直接提供 PriorLabs 身份验证令牌（对于无头/CI 环境很有用）。从 [https://ux.priorlabs.ai](https://ux.priorlabs.ai) 获取一个。
- `TABPFN_NO_BROWSER`：设置为禁用自动基于浏览器的登录（例如在不希望打开浏览器的环境中）。

**模型配置：**
- `TABPFN_MODEL_CACHE_DIR`：自定义目录，用于缓存下载的 TabPFN 模型（默认：平台特定的用户缓存目录）
- `TABPFN_ALLOW_CPU_LARGE_DATASET`：允许在 CPU 上运行大数据集（>1000 样本）的 TabPFN。设置为 `true` 以覆盖 CPU 限制。注意：这会非常慢！

**PyTorch 设置：**
- `PYTORCH_CUDA_ALLOC_CONF`：PyTorch CUDA 内存分配配置，用于优化 GPU 内存使用（默认：`max_split_size_mb:512`）。更多信息请参见 [PyTorch CUDA 文档](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)。

示例：
```bash
export TABPFN_MODEL_CACHE_DIR="/path/to/models"
export TABPFN_ALLOW_CPU_LARGE_DATASET=true
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

或者简单地在 `.env` 中设置它们。

**问：如何保存和加载训练好的 TabPFN 模型？**
答：使用 :func:`save_fitted_tabpfn_model` 持久化拟合的估计器，然后使用 :func:`load_fitted_tabpfn_model` 重新加载（或相应的 ``load_from_fit_state`` 类方法）。

```python
from tabpfn import TabPFNRegressor
from tabpfn.model_loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)

# 在 GPU 上训练回归器
reg = TabPFNRegressor(device="cuda")
reg.fit(X_train, y_train)
save_fitted_tabpfn_model(reg, "my_reg.tabpfn_fit")

# 之后或在仅 CPU 的机器上
reg_cpu = load_fitted_tabpfn_model("my_reg.tabpfn_fit", device="cpu")
```

要仅存储基础模型权重（不存储拟合的估计器），请使用
``save_tabpfn_model(reg.model_, "my_tabpfn.ckpt")``。这仅仅是保存预训练权重的检查点，以便您可以稍后创建和拟合新的估计器。使用 ``load_model_criterion_config`` 重新加载检查点。

### **性能与限制**

**问：TabPFN 能处理缺失值吗？**
答：**可以！**

**问：如何提高 TabPFN 的性能？**
答：最佳实践：
- 使用 [TabPFN Extensions](https://github.com/priorlabs/tabpfn-extensions) 中的 **AutoTabPFNClassifier** 进行训练后集成
- 特征工程：添加特定领域的特征以提高模型性能

无效的做法：
- 自适应特征缩放
- 将分类特征转换为数值（例如独热编码）

**问：HuggingFace 上有哪些不同的检查点？**
答：除了默认检查点外，其他可用的检查点是实验性的，平均表现较差，我们建议始终从默认检查点开始。它们可以用作集成或超参数优化系统的一部分（并自动用于 `AutoTabPFNClassifier`），或手动尝试。它们的名称后缀指的是我们期望它们擅长的方面。

<details>
<summary>每个 TabPFN-2.5 检查点的更多详细信息</summary>

我们在真实数据集上微调的检查点前添加 🌍 表情符号。请参阅 [TabPFN-2.5 论文](https://arxiv.org/abs/2511.08667) 获取 43 个数据集的列表。我们在真实数据集上微调的检查点前添加 🌍 表情符号。请参阅 [TabPFN-2.5 论文](https://arxiv.org/abs/2511.08667) 获取 43 个数据集的列表。我们在真实数据集上微调的检查点前添加 🌍 表情符号。请参阅 [TabPFN-2.5 论文](https://arxiv.org/abs/2511.08667) 获取 43 个数据集的列表。

- `tabpfn-v2.5-classifier-v2.5_default.ckpt` 🌍：默认分类检查点，在真实数据上微调。
- `tabpfn-v2.5-classifier-v2.5_default-2.ckpt`：最佳分类合成检查点。使用此检查点获取不带真实数据微调的默认 TabPFN-2.5 分类模型。
- `tabpfn-v2.5-classifier-v2.5_large-features-L.ckpt`：专为较大特征（最多 500）和小样本（< 5K）设计。
- `tabpfn-v2.5-classifier-v2.5_large-features-XL.ckpt`：专为较大特征（最多 1000，可能支持 `max_features_per_estimator=1000`）设计。
- `tabpfn-v2.5-classifier-v2.5_large-samples.ckpt`：专为较大样本量（大于 30K）设计。
- `tabpfn-v2.5-classifier-v2.5_real.ckpt` 🌍：另一个在真实数据上微调的分类检查点。总体表现相当不错，但在较大特征（>100-200）上表现较差。
- `tabpfn-v2.5-classifier-v2.5_real-large-features.ckpt` 🌍：另一个在真实数据上微调的分类检查点，在大样本（> 10K）上表现较差。
- `tabpfn-v2.5-classifier-v2.5_real-large-samples-and-features.ckpt` 🌍：与 `tabpfn-v2.5-classifier-v2.5_default.ckpt` 相同。
- `tabpfn-v2.5-classifier-v2.5_variant.ckpt`：相当不错，但在较大特征（> 100-200）上表现较差。
- `tabpfn-v2.5-regressor-v2.5_default.ckpt`：默认回归检查点，仅在合成数据上训练。
- `tabpfn-v2.5-regressor-v2.5_low-skew.ckpt`：在低目标偏度数据上专门的变体（但总体表现相当差）。
- `tabpfn-v2.5-regressor-v2.5_quantiles.ckpt`：可能对分位数/分布估计感兴趣的变体，但默认检查点仍应优先考虑。
- `tabpfn-v2.5-regressor-v2.5_real.ckpt` 🌍：在真实数据上微调。在真实数据上微调的检查点中表现最佳。对于回归，我们建议默认使用仅合成的检查点，但此检查点在某些数据集上要好得多。
- `tabpfn-v2.5-regressor-v2.5_real-variant.ckpt` 🌍：另一个在真实数据上微调的回归变体。
- `tabpfn-v2.5-regressor-v2.5_small-samples.ckpt`：在较小（< 3K）样本上略微更好的变体。
- `tabpfn-v2.5-regressor-v2.5_variant.ckpt`：其他变体，没有明确的专长，但在某些数据集上可能更好。

</details>


## 开发

1. 安装 [uv](https://docs.astral.sh/uv/)
2. 设置环境：
```bash
git clone https://github.com/PriorLabs/TabPFN.git
cd TabPFN
uv sync
source venv/bin/activate  # 在 Windows 上：venv\Scripts\activate
pre-commit install
```

3. 提交前：
```bash
pre-commit run --all-files
```

4. 运行测试：
```bash
pytest tests/
```

## 匿名遥测

本项目收集完全匿名的使用遥测数据，并提供选择退出任何遥测或选择加入扩展遥测的选项。

这些数据仅用于帮助我们为相关产品和计算环境提供稳定性，并指导未来改进。

- **不收集任何个人数据**
- **代码、模型输入或输出永远不会被发送**
- **数据严格匿名，无法链接到个人**

有关遥测的详细信息，请参阅我们的[遥测参考](https://github.com/PriorLabs/TabPFN/blob/main/TELEMETRY.md)和我们的[隐私政策](https://priorlabs.ai/privacy-policy/)。

**要选择退出**，请设置以下环境变量：

```bash
export TABPFN_DISABLE_TELEMETRY=1
```
---

由 ❤️ 构建于 [Prior Labs](https://priorlabs.ai) - 版权所有 (c) 2026 Prior Labs GmbH