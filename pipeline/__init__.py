"""
Modular Hierarchical Time Series Forecasting Pipeline

Clean, focused modules for each step of the forecasting process:
- DataLoader: Load and combine data sources
- DataSplitter: Create train/test splits 
- FeatureEngineer: Handle feature engineering
- Forecaster: Train models and create forecasts
- Evaluator: Run hierarchical evaluation
- ForecastingPipeline: Orchestrate all components
"""

from .data_loader import DataLoader
from .data_splitter import DataSplitter
from .feature_engineer import FeatureEngineer
from .forecaster import Forecaster
from .evaluator import Evaluator
from .pipeline_orchestrator import ForecastingPipeline

__version__ = "1.0.0"
__author__ = "Forecasting Team"

__all__ = [
    'DataLoader',
    'DataSplitter', 
    'FeatureEngineer',
    'Forecaster',
    'Evaluator',
    'ForecastingPipeline'
] 