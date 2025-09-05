# Hierarchical Time Series Forecasting Dashboard

Streamlit dashboard for exploring pre-computed hierarchical forecasting results with interactive visualizations.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard available at `http://localhost:8501`

## 📁 Structure

```
lyra/
├── app.py              # Streamlit dashboard
├── data/               # Forecast data & evaluations
└── requirements.txt    # Dependencies
```

## 📊 Features

### 🔍 Validation Mode
- Analyze pre-computed forecast results
- Compare 5 models: TimeGPT, LGB, XGB, RF, Ensemble
- View performance across 8 hierarchy levels

### 🚀 Forecasting Mode  
- Explore latest forecast results
- Drill down from overall to specific combinations
- Weekly/monthly aggregation options

## 📈 Data Requirements

Place these files in `data/` directory:
- `all_bottom_forecast_*.parquet` - Forecast results
- `evaluation_*/` - Performance metrics
- `feature_importance_*.json` - Model features

## 🎯 Hierarchy Levels

1. Overall
2. Company / State / Program  
3. Company+State / Company+Program / State+Program
4. Company+State+Program

Dashboard auto-detects available periods and displays champion/challenger models with comprehensive metrics (MAE, MSE, RMSE, MAPE, WMAPE).