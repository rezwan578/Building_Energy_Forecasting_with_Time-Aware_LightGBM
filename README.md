# Building Energy Forecasting with Time-Aware LightGBM
An applied machine learning project for predicting building-level energy consumption using the ASHRAE Great Energy Predictor III dataset. The project focuses on large-scale tabular regression, time-aware feature engineering, chronological model evaluation, LightGBM modelling, interpretability, and meter-wise error analysis.

## Project Context

This project was developed for the Applied Machine Learning module within the MSc Data Science and Analytics programme at Cardiff University, United Kingdom.

Author: Rezwan Ahmed/
Supervisor: Yingying Wu

This repository presents a cleaned, portfolio-ready version of the original academic project. The main analysis is provided as a single Jupyter/Google Colab notebook to preserve the exploratory and experimental workflow.

## Objective

Building energy consumption is influenced by building characteristics, meter type, site-level weather, operational patterns, and temporal effects. The objective of this project is to develop a compact and interpretable machine learning pipeline for predicting energy usage from timestamped meter readings, building metadata, and weather observations.

The task is formulated as a supervised regression problem using the log-transformed meter reading as the modelling target.

## Dataset

The project uses the ASHRAE Great Energy Predictor III dataset, which contains hourly energy meter readings from non-residential buildings, along with building metadata and site-level weather observations.

The raw dataset is not included in this repository because of size and licensing considerations. The dataset should be obtained directly from Kaggle.

Main input files used in the analysis:

```text
train.csv
building_metadata.csv
weather_train.csv
```

## Methodology

The workflow includes:

- Data integration across meter readings, building metadata, and weather observations
- Missing-value analysis and feature removal
- Memory optimisation for large-scale tabular data
- Domain-specific target correction
- Log transformation of the target variable
- Time-aware feature engineering
- Chronological train-validation-test splitting
- Model comparison across linear and tree-based methods
- LightGBM tuning
- SHAP-based interpretability analysis
- Meter-wise error analysis

## Feature Engineering

The project includes several domain-informed feature engineering steps:

- Log transformation of `meter_reading`
- Log transformation of `square_feet`
- Extraction of temporal features from timestamps:
  - Hour
  - Weekday
  - Month
  - Week of year
  - Weekend indicator
  - Working-time indicator
- Temperature-time interaction features
- Model-specific handling of categorical variables

## Models Compared

The following models were evaluated:

- Linear Regression
- Ridge Regression
- Random Forest
- LightGBM base model
- Fine-tuned LightGBM with `building_id`
- Fine-tuned LightGBM without `building_id`

A chronological evaluation strategy was used instead of a random split to better reflect future-period prediction.

## Results

All errors are reported on the log-transformed target scale.

| Model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| Linear Regression | 1.807 | 1.296 | 0.251 |
| Ridge Regression | 1.806 | 1.296 | 0.252 |
| Random Forest | 1.210 | 0.751 | 0.666 |
| LightGBM Base Model | 1.198 | 0.696 | 0.670 |
| Fine-tuned LightGBM without `building_id` | 1.184 | 0.690 | 0.680 |
| Fine-tuned LightGBM with `building_id` | 1.181 | 0.641 | 0.680 |

The preferred final model was the fine-tuned LightGBM model without `building_id`. Although including `building_id` slightly improved some error metrics, removing it reduced reliance on building-specific identity information and shifted the model toward more generalisable predictors. Since it was trained and evaluated on log-transformed meter readings, its RMSE of 1.184 is on the same logarithmic scale as the RMSLE metric used in the ASHRAE GEPIII competition. The result is competitive relative to the reported winning RMSLE of 1.231, but it should not be interpreted as a direct leaderboard comparison because the official competition used a hidden future test set. The safer conclusion is that a compact, carefully engineered, single-model LightGBM pipeline produced a robust and competitive local evaluation result.

## Interpretability

SHAP analysis was used to examine model behaviour.

When `building_id` was included, the model relied strongly on building identity. After removing `building_id`, feature importance shifted toward more interpretable and generalisable predictors, including:

- Log-transformed building size
- Site ID
- Meter type
- Primary use
- Week of year
- Air temperature
- Hour
- Month

This supported the selection of the LightGBM model without `building_id` as the preferred final model.

## Meter-Wise Error Analysis

The final model was also evaluated separately across meter categories. Electricity showed the strongest performance, while hot water had the highest prediction error. This suggests that model performance can vary across energy categories and should not be interpreted only through aggregate metrics.

## Repository Structure

```text
building-energy-forecasting-lightgbm/
│
├── README.md
│
├── notebook/
│   └── energy_usage_prediction_lightgbm.ipynb
│
├── reports/
│   └── technical_report.pdf
```

## Notebook

The main analysis is provided in a single notebook:

```text
notebook/energy_usage_prediction_lightgbm.ipynb
```

The notebook contains the full workflow, including data loading, preprocessing, feature engineering, model training, evaluation, and interpretability analysis.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- LightGBM
- SHAP
- Google Colab

## Limitations

The chronological split evaluates future-period prediction for known buildings. It does not fully test generalisation to unseen buildings or unseen sites. Future work should explore:

- Building-level holdout validation
- Site-level holdout validation
- Improved weather timestamp alignment
- Occupancy and schedule-based features
- Equipment-level data
- Sequential models for time-dependent prediction


## Status

This repository is intended as a research-style machine learning portfolio project. The focus is on model development, time-aware evaluation, interpretability, and clear communication rather than leaderboard optimisation.
