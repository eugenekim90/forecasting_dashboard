import pandas as pd
import numpy as np
import os
from helper import series_metrics

def generate_evaluation_files(split_date, horizon):
   
    # Load forecast data
    forecast_file = f"data/all_bottom_forecast_{split_date}_{horizon}.parquet"
    if not os.path.exists(forecast_file):
        print(f"Error: {forecast_file} not found!")
        return
    
    print(f"Loading forecast data from {forecast_file}")
    df = pd.read_parquet(forecast_file)
    df['ds'] = pd.to_datetime(df['ds'])
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    # Create evaluation directory
    eval_dir = f"data/evaluation_{split_date}_{horizon}"
    os.makedirs(eval_dir, exist_ok=True)
    
    models = ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML']
    
    # === 1. COMPANY_PROGRAM_STATE LEVEL (Bottom level - most granular) ===
    # Weekly
    print("Generating weekly company-program-state evaluation...")
    bottom_evaluation_weekly = series_metrics(df, horizon=int(horizon), models=models)
    bottom_evaluation_weekly.to_parquet(f"{eval_dir}/weekly_company_program_state_evaluation.parquet", index=False)
    
    # Monthly
    print("Generating monthly company-program-state evaluation...")
    df_monthly = df.groupby(['unique_id', pd.Grouper(key='ds', freq='ME')])[
        ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']
    ].sum().reset_index()
    bottom_evaluation_monthly = series_metrics(df_monthly, horizon=int(horizon), models=models)
    bottom_evaluation_monthly.to_parquet(f"{eval_dir}/monthly_company_program_state_evaluation.parquet", index=False)
    
    # === 2. COMPANY_PROGRAM LEVEL ===
    # Weekly
    print("Generating weekly company-program evaluation...")
    company_program_weekly = df.groupby(['company', 'program', 'ds']).agg({
        'TimeGPT': 'sum', 'LGB': 'sum', 'XGB': 'sum', 'RF': 'sum', 'ensemble': 'sum', 'ensemble_ML': 'sum', 'y': 'sum'
    }).reset_index()
    company_program_weekly['unique_id'] = company_program_weekly['company'] + '_' + company_program_weekly['program']
    company_program_weekly_evaluation = series_metrics(company_program_weekly, horizon=int(horizon), models=models)
    company_program_weekly_evaluation.to_parquet(f"{eval_dir}/weekly_company_program_evaluation.parquet", index=False)
    
    # Monthly
    print("Generating monthly company-program evaluation...")
    company_program_monthly = df.groupby(['company', 'program', pd.Grouper(key='ds', freq='ME')])[
        ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']
    ].sum().reset_index()
    company_program_monthly['unique_id'] = company_program_monthly['company'] + '_' + company_program_monthly['program']
    company_program_monthly_evaluation = series_metrics(company_program_monthly, horizon=int(horizon), models=models)
    company_program_monthly_evaluation.to_parquet(f"{eval_dir}/monthly_company_program_evaluation.parquet", index=False)
    
    # === 3. COMPANY_STATE LEVEL ===
    # Weekly
    print("Generating weekly company-state evaluation...")
    company_state_weekly = df.groupby(['company', 'state', 'ds']).agg({
        'TimeGPT': 'sum', 'LGB': 'sum', 'XGB': 'sum', 'RF': 'sum', 'ensemble': 'sum', 'ensemble_ML': 'sum', 'y': 'sum'
    }).reset_index()
    company_state_weekly['unique_id'] = company_state_weekly['company'] + '_' + company_state_weekly['state']
    company_state_weekly_evaluation = series_metrics(company_state_weekly, horizon=int(horizon), models=models)
    company_state_weekly_evaluation.to_parquet(f"{eval_dir}/weekly_company_state_evaluation.parquet", index=False)
    
    # Monthly
    print("Generating monthly company-state evaluation...")
    company_state_monthly = df.groupby(['company', 'state', pd.Grouper(key='ds', freq='ME')])[
        ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']
    ].sum().reset_index()
    company_state_monthly['unique_id'] = company_state_monthly['company'] + '_' + company_state_monthly['state']
    company_state_monthly_evaluation = series_metrics(company_state_monthly, horizon=int(horizon), models=models)
    company_state_monthly_evaluation.to_parquet(f"{eval_dir}/monthly_company_state_evaluation.parquet", index=False)
    
    # === 4. PROGRAM_STATE LEVEL ===
    # Weekly
    print("Generating weekly program-state evaluation...")
    program_state_weekly = df.groupby(['program', 'state', 'ds']).agg({
        'TimeGPT': 'sum', 'LGB': 'sum', 'XGB': 'sum', 'RF': 'sum', 'ensemble': 'sum', 'ensemble_ML': 'sum', 'y': 'sum'
    }).reset_index()
    program_state_weekly['unique_id'] = program_state_weekly['program'] + '_' + program_state_weekly['state']
    program_state_weekly_evaluation = series_metrics(program_state_weekly, horizon=int(horizon), models=models)
    program_state_weekly_evaluation.to_parquet(f"{eval_dir}/weekly_program_state_evaluation.parquet", index=False)
    
    # Monthly
    print("Generating monthly program-state evaluation...")
    program_state_monthly = df.groupby(['program', 'state', pd.Grouper(key='ds', freq='ME')])[
        ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']
    ].sum().reset_index()
    program_state_monthly['unique_id'] = program_state_monthly['program'] + '_' + program_state_monthly['state']
    program_state_monthly_evaluation = series_metrics(program_state_monthly, horizon=int(horizon), models=models)
    program_state_monthly_evaluation.to_parquet(f"{eval_dir}/monthly_program_state_evaluation.parquet", index=False)
    
    # === 5. COMPANY LEVEL ===
    # Weekly
    print("Generating weekly company evaluation...")
    company_weekly = df.groupby(['company', 'ds']).agg({
        'TimeGPT': 'sum', 'LGB': 'sum', 'XGB': 'sum', 'RF': 'sum', 'ensemble': 'sum', 'ensemble_ML': 'sum', 'y': 'sum'
    }).reset_index()
    company_weekly['unique_id'] = company_weekly['company']
    company_weekly_evaluation = series_metrics(company_weekly, horizon=int(horizon), models=models)
    company_weekly_evaluation.to_parquet(f"{eval_dir}/weekly_company_evaluation.parquet", index=False)
    
    # Monthly
    print("Generating monthly company evaluation...")
    company_monthly = df.groupby(['company', pd.Grouper(key='ds', freq='ME')])[
        ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']
    ].sum().reset_index()
    company_monthly['unique_id'] = company_monthly['company']
    company_monthly_evaluation = series_metrics(company_monthly, horizon=int(horizon), models=models)
    company_monthly_evaluation.to_parquet(f"{eval_dir}/monthly_company_evaluation.parquet", index=False)
    
    # === 6. PROGRAM LEVEL ===
    # Weekly
    print("Generating weekly program evaluation...")
    program_weekly = df.groupby(['program', 'ds']).agg({
        'TimeGPT': 'sum', 'LGB': 'sum', 'XGB': 'sum', 'RF': 'sum', 'ensemble': 'sum', 'ensemble_ML': 'sum', 'y': 'sum'
    }).reset_index()
    program_weekly['unique_id'] = program_weekly['program']
    program_weekly_evaluation = series_metrics(program_weekly, horizon=int(horizon), models=models)
    program_weekly_evaluation.to_parquet(f"{eval_dir}/weekly_program_evaluation.parquet", index=False)
    
    # Monthly
    print("Generating monthly program evaluation...")
    program_monthly = df.groupby(['program', pd.Grouper(key='ds', freq='ME')])[
        ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']
    ].sum().reset_index()
    program_monthly['unique_id'] = program_monthly['program']
    program_monthly_evaluation = series_metrics(program_monthly, horizon=int(horizon), models=models)
    program_monthly_evaluation.to_parquet(f"{eval_dir}/monthly_program_evaluation.parquet", index=False)
    
    # === 7. STATE LEVEL ===
    # Weekly
    print("Generating weekly state evaluation...")
    state_weekly = df.groupby(['state', 'ds']).agg({
        'TimeGPT': 'sum', 'LGB': 'sum', 'XGB': 'sum', 'RF': 'sum', 'ensemble': 'sum', 'ensemble_ML': 'sum', 'y': 'sum'
    }).reset_index()
    state_weekly['unique_id'] = state_weekly['state']
    state_weekly_evaluation = series_metrics(state_weekly, horizon=int(horizon), models=models)
    state_weekly_evaluation.to_parquet(f"{eval_dir}/weekly_state_evaluation.parquet", index=False)
    
    # Monthly
    print("Generating monthly state evaluation...")
    state_monthly = df.groupby(['state', pd.Grouper(key='ds', freq='ME')])[
        ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']
    ].sum().reset_index()
    state_monthly['unique_id'] = state_monthly['state']
    state_monthly_evaluation = series_metrics(state_monthly, horizon=int(horizon), models=models)
    state_monthly_evaluation.to_parquet(f"{eval_dir}/monthly_state_evaluation.parquet", index=False)
    
    # === 8. OVERALL LEVEL ===
    # Weekly
    print("Generating weekly overall evaluation...")
    overall_weekly = df.groupby(['ds']).agg({
        'TimeGPT': 'sum', 'LGB': 'sum', 'XGB': 'sum', 'RF': 'sum', 'ensemble': 'sum', 'ensemble_ML': 'sum', 'y': 'sum'
    }).reset_index()
    overall_weekly['unique_id'] = "overall"
    overall_weekly_evaluation = series_metrics(overall_weekly, horizon=int(horizon), models=models)
    overall_weekly_evaluation.to_parquet(f"{eval_dir}/weekly_overall_evaluation.parquet", index=False)
    
    # Monthly
    print("Generating monthly overall evaluation...")
    overall_monthly = df.groupby(pd.Grouper(key='ds', freq='ME')).agg({
        'TimeGPT': 'sum', 'LGB': 'sum', 'XGB': 'sum', 'RF': 'sum', 'ensemble': 'sum', 'ensemble_ML': 'sum', 'y': 'sum'
    }).reset_index()
    overall_monthly['unique_id'] = "overall"
    overall_monthly_evaluation = series_metrics(overall_monthly, horizon=int(horizon), models=models)
    overall_monthly_evaluation.to_parquet(f"{eval_dir}/monthly_overall_evaluation.parquet", index=False)
    
    # === 9. GENERATE SUMMARY FILES ===
    print("Generating summary files...")
    
    # Collect all evaluation objects for summary tables
    levels_weekly = {
        'overall': overall_weekly_evaluation,
        'company': company_weekly_evaluation,
        'program': program_weekly_evaluation,
        'state': state_weekly_evaluation,
        'company_program': company_program_weekly_evaluation,
        'company_state': company_state_weekly_evaluation,
        'program_state': program_state_weekly_evaluation,
        'company_program_state': bottom_evaluation_weekly,
    }
    
    levels_monthly = {
        'overall': overall_monthly_evaluation,
        'company': company_monthly_evaluation,
        'program': program_monthly_evaluation,
        'state': state_monthly_evaluation,
        'company_program': company_program_monthly_evaluation,
        'company_state': company_state_monthly_evaluation,
        'program_state': program_state_monthly_evaluation,
        'company_program_state': bottom_evaluation_monthly,
    }
    
    metrics = ['MAE', 'RMSE', 'MAPE', 'WMAPE']
    
    def build_tables(levels):
        """Build summary tables using mean (matching test.py logic)"""
        tables = {}
        for m in metrics:
            data = {
                lvl: [ev[f'{m}_{mdl}'].mean() for mdl in models]
                for lvl, ev in levels.items()
            }
            df = pd.DataFrame(data, index=models)
            tables[m] = df.round(2)
        return tables
    
    # Generate summary tables
    weekly_tables = build_tables(levels_weekly)
    monthly_tables = build_tables(levels_monthly)
    
    # Save summary files
    for metric in metrics:
        weekly_tables[metric].to_parquet(f"{eval_dir}/weekly_{metric.lower()}_summary.parquet")
        monthly_tables[metric].to_parquet(f"{eval_dir}/monthly_{metric.lower()}_summary.parquet")
    
    print(f"✅ Evaluation files generated successfully for {split_date}_{horizon}")
    print(f"📁 Files saved in: {eval_dir}/")

