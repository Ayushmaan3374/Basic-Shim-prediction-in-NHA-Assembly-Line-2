# 🔩 BASICSHIM Prediction for NHA Assembly Line

## 📌 Project Overview
In precision manufacturing, maintaining component alignment during assembly is crucial for product quality and consistency. The BASICSHIM value serves as a correction factor applied during part fitting, based on left-hand and right-hand shim measurements.

This project aims to automate the prediction of BASICSHIM values using machine learning, thereby reducing manual calibration effort and increasing throughput on production lines.

## 🎯 Objective
Build a robust regression model that accurately predicts BASICSHIM from:
- `LHSSHIM`: Left-Hand Side Shim
- `RHSSHIM`: Right-Hand Side Shim
- `TYPE`: Specific part type or configuration

## 🧠 Approach
A variety of ML models were evaluated, focusing on their ability to:
- Capture nonlinear patterns across diverse configurations
- Generalize across multiple component types
- Maintain high prediction accuracy with minimal latency

After comparative analysis, **LightGBM** was selected as the final model due to:
- ⚡ High accuracy across test cases
- 🚀 Fast training and inference speed
- 📈 Scalable deployment capabilities

## 📊 Results
- Significant reduction in manual calibration time
- Improved prediction consistency across different part types
- Streamlined production with data-driven automation

## 🔧 Tech Stack
- Python (NumPy, pandas, scikit-learn)
- LightGBM
- Joblib (for model serialization)

## 📁 Repository Structure
