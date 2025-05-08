# Energy Consumption Forecasting Project

This repository contains various time series forecasting models for energy consumption prediction. The project explores different approaches ranging from traditional statistical methods to advanced machine learning and deep learning techniques.

## Project Overview

The main objective of this project is to develop accurate forecasting models for energy consumption data. The models are designed to capture both linear and non-linear patterns in energy usage over time, considering seasonal variations and external factors like temperature.

## Key Achievement: Hybrid Model

The primary achievement of this project is the development of a **hybrid forecasting model** (10_gibrid_model.py) that combines multiple approaches to leverage their individual strengths:

- Statistical time series components from ARIMA/SARIMA models
- Machine learning capabilities from XGBoost/Random Forest
- Deep learning patterns from LSTM/GRU networks

The hybrid approach demonstrates superior performance by capturing both linear and non-linear relationships in the data, handling seasonality, and incorporating exogenous variables effectively.

## Models Implemented

### Statistical Models
- **1_ARIMA.py**: Autoregressive Integrated Moving Average model
- **2_SARIMA.py**: Seasonal ARIMA model for capturing seasonal patterns
- **3_SARIMAX.py**: SARIMA with exogenous variables
- **3_SARIMAX_HARD_version.py**: Advanced implementation of SARIMAX
- **3_1_SARIMAX_Easier_version.py**: Simplified SARIMAX implementation
- **3_1_SARIMAX_with_temperature.py**: SARIMAX incorporating temperature data
- **3_2_SARIMAX_normalized.py**: SARIMAX with normalized data

### Machine Learning Models
- **4_Linear_regression_model.py**: Basic linear regression approach
- **5_RF_model.py**: Random Forest model for time series forecasting
- **6_XGBoost_model.py**: XGBoost implementation for forecasting
- **6_XGBoost_model_forecast.py**: XGBoost with specific forecast implementation
- **6_XGBoost_model_forecast_v2.py**: Improved version of XGBoost forecast
- **6_XGBoost_model_month.py**: XGBoost adjusted for monthly forecasting
- **6_XGBoost_without_lags.py**: XGBoost implementation without lag features

### Deep Learning Models
- **7_LightGBM_model.py**: LightGBM implementation for forecasting
- **8_GRU_model.py**: Gated Recurrent Unit neural network model
- **9_LSTM_Model.py**: Long Short-Term Memory neural network model

### Hybrid Model
- **10_gibrid_model.py**: The flagship hybrid model combining multiple approaches

## Features

- Time series decomposition and analysis
- Feature engineering for energy consumption data
- Handling of seasonal patterns and trends
- Incorporation of exogenous variables (e.g., temperature)
- Model evaluation and comparison framework
- Hyperparameter tuning for optimal performance
- Forecasting at different time horizons (day, month)

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, statsmodels, xgboost, lightgbm, tensorflow/keras

### Installation
```bash
git clone https://github.com/yourusername/energy-consumption-forecasting.git
cd energy-consumption-forecasting
pip install -r requirements.txt
