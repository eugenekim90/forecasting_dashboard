import pandas as pd
import numpy as np
import glob
import os
from interval import per_model_intervals_no_nan

def add_intervals_to_latest_forecast():
    """
    Add prediction intervals to the latest forecast file
    """
    print("🔧 Adding prediction intervals to latest forecast...")
    
    # Load latest forecast
    latest_file = 'data/all_bottom_forecast_latest.parquet'
    if not os.path.exists(latest_file):
        print(f"❌ Error: {latest_file} not found!")
        return
    
    print(f"📂 Loading {latest_file}")
    df = pd.read_parquet(latest_file)
    print(f"📊 Data shape: {df.shape}")
    
    # Try to find validation data for calibration
    validation_df = None
    try:
        validation_files = glob.glob('data/all_bottom_forecast_*.parquet')
        validation_files = [f for f in validation_files if 'latest' not in f]
        if validation_files:
            latest_validation = max(validation_files, key=lambda x: os.path.getmtime(x))
            temp_df = pd.read_parquet(latest_validation)
            if 'y' in temp_df.columns:
                validation_df = temp_df
                print(f"✅ Found validation data: {os.path.basename(latest_validation)}")
            else:
                print("⚠️ No 'y' column in validation files")
    except Exception as e:
        print(f"⚠️ Could not load validation data: {e}")
    
    # Get available models
    model_cols = ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble']
    if 'ensemble_ML' in df.columns:
        model_cols.append('ensemble_ML')
    
    available_models = [col for col in model_cols if col in df.columns]
    print(f"🤖 Available models: {available_models}")
    
    if validation_df is not None and 'y' in validation_df.columns:
        print("🎯 Computing intervals using validation data calibration...")
        
        # Get intervals from validation data to calibrate widths
        intervals_cal = per_model_intervals_no_nan(
            validation_df,
            model_cols=available_models,
            alpha=0.10,  # 90% intervals
            nonnegative=True,
            zeros_as_missing=False,
            fallback_rel=0.30,
            fallback_abs=0.1,
            fillna_yhat_with=0.0
        )
        
        # Extract interval widths per series and model
        interval_widths = {}
        for model in available_models:
            if f'{model}_lo' in intervals_cal.columns and f'{model}_hi' in intervals_cal.columns:
                intervals_cal[f'{model}_width'] = (intervals_cal[f'{model}_hi'] - intervals_cal[f'{model}_lo']) / 2
                # Get median width per unique_id for this model
                width_by_series = intervals_cal.groupby('unique_id')[f'{model}_width'].median().to_dict()
                interval_widths[model] = width_by_series
        
        # Apply intervals to forecast data
        print("📈 Applying calibrated intervals to forecast data...")
        df_with_intervals = df.copy()
        
        for model in available_models:
            if model in interval_widths:
                # Map interval widths to forecast data
                df_with_intervals[f'{model}_width'] = df_with_intervals['unique_id'].map(
                    interval_widths[model]
                ).fillna(
                    # Fallback: use global median or relative width
                    df_with_intervals[model].abs() * 0.30
                )
                
                # Generate intervals
                df_with_intervals[f'{model}_lo'] = np.maximum(
                    0.0,  # Non-negative constraint
                    df_with_intervals[model] - df_with_intervals[f'{model}_width']
                )
                df_with_intervals[f'{model}_hi'] = (
                    df_with_intervals[model] + df_with_intervals[f'{model}_width']
                )
                
                # Drop the width column
                df_with_intervals = df_with_intervals.drop(columns=[f'{model}_width'])
                
        print("✅ Calibrated intervals applied")
    
    else:
        print("🔄 Using fallback relative intervals...")
        df_with_intervals = df.copy()
        
        for model in available_models:
            # Use 30% relative width as fallback
            width = np.maximum(0.1, df[model].abs() * 0.30)
            df_with_intervals[f'{model}_lo'] = np.maximum(0.0, df[model] - width)
            df_with_intervals[f'{model}_hi'] = df[model] + width
        
        print("✅ Fallback intervals applied")
    
    # Save the updated file
    output_file = latest_file  # Overwrite the original
    print(f"💾 Saving updated forecast with intervals to {output_file}")
    df_with_intervals.to_parquet(output_file, index=False)
    
    print("✅ Prediction intervals successfully added to forecast file!")
    print(f"📊 New columns added: {[col for col in df_with_intervals.columns if '_lo' in col or '_hi' in col]}")
    print(f"📈 File size: {df_with_intervals.shape}")
    
    return df_with_intervals

if __name__ == "__main__":
    result = add_intervals_to_latest_forecast() 