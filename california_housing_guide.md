# 🏡 TabPFN Regression Guide: California Housing Dataset

## 📌 Overview

This tutorial provides a complete, step-by-step guide on training the `TabPFNRegressor` model using the **California Housing dataset**. It addresses issue [#293](https://github.com/PriorLabs/TabPFN/issues/293) by demonstrating:

- How to load and preprocess real-world tabular data
- How to fully train TabPFN for regression
- How to evaluate model performance (R², MSE)
- How to visualize predictions and errors

By the end of this guide, you will have a working example of TabPFN on a popular regression dataset—with insights, performance metrics, and clear code examples.

---

## ⚙️ Setup Instructions

1. Clone the repository and create a virtual environment:
   ```bash
   git clone https://github.com/your-username/TabPFN.git
   cd TabPFN
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
