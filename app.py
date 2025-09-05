import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def load_feature_importance(split_date, horizon):
    """Load feature importance data for a specific period"""
    import json
    
    feature_file = f"data/feature_importance_{split_date}_{horizon}.json"
    
    try:
        with open(feature_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return None if no feature importance data available
        return None
    except Exception as e:
        st.warning(f"Error loading feature importance: {e}")
        return None

def get_champion_challenger(metrics_df):
    """Get champion (best) and challenger (second best) models based on RMSE"""
    if 'RMSE' not in metrics_df.columns or len(metrics_df) < 2:
        return None, None
    
    # Sort by RMSE to get best and second best
    sorted_df = metrics_df.sort_values('RMSE')
    
    champion_idx = sorted_df.index[0]
    challenger_idx = sorted_df.index[1] if len(sorted_df) > 1 else sorted_df.index[0]
    
    champion = {
        'model': sorted_df.loc[champion_idx, 'Model'],
        'rmse': sorted_df.loc[champion_idx, 'RMSE']
    }
    
    challenger = {
        'model': sorted_df.loc[challenger_idx, 'Model'],
        'rmse': sorted_df.loc[challenger_idx, 'RMSE']
    }
    
    return champion, challenger

# Helper functions - moved to top to avoid NameError
def get_series_display_name(selected_level, selected_company, selected_state, selected_program, forecast_data):
    """Generate clean display name for the series"""
    if selected_level == "Overall":
        return "Overall Level (All Data)"
    elif selected_level == "Company":
        if selected_company:
            return f"Company Level - {selected_company}"
        else:
            first_company = sorted(forecast_data['company'].unique())[0]
            return f"Company Level - {first_company} (Sample)"
    elif selected_level == "State":
        if selected_state:
            return f"State Level - {selected_state}"
        else:
            first_state = sorted(forecast_data['state'].unique())[0]
            return f"State Level - {first_state} (Sample)"
    elif selected_level == "Program":
        if selected_program:
            return f"Program Level - {selected_program}"
        else:
            first_program = sorted(forecast_data['program'].unique())[0]
            return f"Program Level - {first_program} (Sample)"
    elif selected_level == "Company + State":
        comp_part = selected_company or f"{sorted(forecast_data['company'].unique())[0]} (Sample)"
        state_part = selected_state or f"{sorted(forecast_data['state'].unique())[0]} (Sample)"
        return f"Company+State Level - {comp_part} × {state_part}"
    elif selected_level == "Company + Program":
        comp_part = selected_company or f"{sorted(forecast_data['company'].unique())[0]} (Sample)"
        prog_part = selected_program or f"{sorted(forecast_data['program'].unique())[0]} (Sample)"
        return f"Company+Program Level - {comp_part} × {prog_part}"
    elif selected_level == "State + Program":
        state_part = selected_state or f"{sorted(forecast_data['state'].unique())[0]} (Sample)"
        prog_part = selected_program or f"{sorted(forecast_data['program'].unique())[0]} (Sample)"
        return f"State+Program Level - {state_part} × {prog_part}"
    elif selected_level == "Company + State + Program":
        comp_part = selected_company or f"{sorted(forecast_data['company'].unique())[0]} (Sample)"
        state_part = selected_state or f"{sorted(forecast_data['state'].unique())[0]} (Sample)"
        prog_part = selected_program or f"{sorted(forecast_data['program'].unique())[0]} (Sample)"
        return f"Company+State+Program Level - {comp_part} × {state_part} × {prog_part}"
    else:
        return selected_level

def get_data_source_info(df, forecast_file_exists):
    unique_series = df['unique_id'].nunique()
    date_range = f"{df['ds'].min().strftime('%Y-%m-%d')} to {df['ds'].max().strftime('%Y-%m-%d')}"
    
    if forecast_file_exists:
        source = "Pre-computed forecast file"
        icon = "📁"
    else:
        source = "Real-time pipeline execution"
        icon = "⚡"
    
    return source, icon, unique_series, date_range

def get_available_periods():
    """Get available forecast periods from data directory"""
    periods = []
    data_dir = "data"
    
    # Look for forecast files
    forecast_files = glob.glob(f"{data_dir}/all_bottom_forecast_*.parquet")
    forecast_files.extend(glob.glob(f"{data_dir}/forecast_*.parquet"))
    
    for file in forecast_files:
        filename = os.path.basename(file)
        if 'all_bottom_forecast_' in filename:
            parts = filename.replace('all_bottom_forecast_', '').replace('.parquet', '').split('_')
        elif 'forecast_' in filename:
            parts = filename.replace('forecast_', '').replace('.parquet', '').split('_')
        else:
            continue
            
        if len(parts) >= 2:
            date_part = parts[0]
            horizon = parts[1]
            
            # Convert date format if needed
            try:
                if len(date_part.split('-')) == 3:
                    date_obj = datetime.strptime(date_part, '%Y-%m-%d')
                    end_date = date_obj + timedelta(weeks=int(horizon))
                    display_text = f"{date_part} to {end_date.strftime('%Y-%m-%d')} ({horizon}w)"
                    periods.append((date_part, horizon, display_text))
            except:
                continue
    
    return sorted(list(set(periods)), key=lambda x: x[0], reverse=True)

def display_fallback_metrics(metrics_df):
    """Display fallback metrics with consistent styling"""
    if not metrics_df.empty:
        st.dataframe(
            metrics_df,
            width="stretch",
            hide_index=True
        )
        
        # Show best model
        if 'RMSE' in metrics_df.columns:
            best_model_idx = metrics_df['RMSE'].idxmin()
            best_model = metrics_df.loc[best_model_idx, 'Model']
            best_rmse = metrics_df.loc[best_model_idx, 'RMSE']
            
            st.markdown("### 🏆 Best Model")
            st.metric(
                label="Lowest RMSE",
                value=best_model,
                delta=f"RMSE: {best_rmse}"
            )
        
    else:
        st.warning("Could not calculate metrics")

def filter_evaluation_data(eval_df, selected_company, selected_state, selected_program):
    """Filter evaluation data based on selections with proper handling"""
    if eval_df.empty:
        return eval_df
    
    filtered_df = eval_df.copy()
    
    # Apply filters only if the column exists and selection is made
    if selected_company and 'company' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['company'] == selected_company]
    
    if selected_state and 'state' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['state'] == selected_state]
    
    if selected_program and 'program' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['program'] == selected_program]
    
    return filtered_df

def load_original_evaluation(split_date, horizon, level, frequency='weekly'):
    """Load original evaluation files directly for exact number matching"""
    eval_dir = f"data/evaluation_{split_date}_{horizon}"
    
    level_mapping = {
        'Overall': 'overall',
        'Company': 'company', 
        'State': 'state',
        'Program': 'program',
        'Company + State': 'company_state',
        'Company + Program': 'company_program',
        'State + Program': 'program_state',
        'Company + State + Program': 'company_program_state'
    }
    
    level_key = level_mapping.get(level)
    if not level_key:
        return pd.DataFrame()
    
    file_path = f"{eval_dir}/{frequency}_{level_key}_evaluation.parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    
    return pd.DataFrame()

def filter_original_evaluation(eval_df, selected_level, selected_company, selected_state, selected_program):
    """Filter original evaluation files based on selections"""
    if eval_df.empty:
        return eval_df
    
    filtered_df = eval_df.copy()
    
    # For different levels, filter by the appropriate unique_id pattern
    # Based on actual patterns found in evaluation files:
    
    if selected_level == "Company":
        if selected_company:
            # Company level: unique_id is just the company name (e.g., "acorabloomcorp")
            filtered_df = filtered_df[filtered_df['unique_id'] == selected_company]
        # If no specific company selected (All), show all companies (no filtering)
        
    elif selected_level == "State":
        if selected_state:
            # State level: unique_id is just the state code (e.g., "CA")  
            filtered_df = filtered_df[filtered_df['unique_id'] == selected_state]
        # If no specific state selected (All), show all states (no filtering)
        
    elif selected_level == "Program":
        if selected_program:
            # Program level: unique_id is just the program name (e.g., "clinical_leave_evaluation")
            filtered_df = filtered_df[filtered_df['unique_id'] == selected_program]
        # If no specific program selected (All), show all programs (no filtering)
        
    elif selected_level == "Company + State":
        if selected_company and selected_state:
            # Both selected: unique_id is "company_state" (e.g., "acorabloomcorp_CA")
            target_id = f"{selected_company}_{selected_state}"
            filtered_df = filtered_df[filtered_df['unique_id'] == target_id]
        elif selected_company:
            # Only company selected: show all states for this company
            filtered_df = filtered_df[filtered_df['unique_id'].str.startswith(f"{selected_company}_")]
        elif selected_state:
            # Only state selected: show all companies for this state  
            filtered_df = filtered_df[filtered_df['unique_id'].str.endswith(f"_{selected_state}")]
        # If both are "All" (neither selected), show all company-state combinations (no filtering)
        
    elif selected_level == "Company + Program":
        if selected_company and selected_program:
            # Both selected: unique_id is "company_program" (e.g., "acorabloomcorp_clinical_leave_evaluation")
            target_id = f"{selected_company}_{selected_program}"
            filtered_df = filtered_df[filtered_df['unique_id'] == target_id]
        elif selected_company:
            # Only company selected: show all programs for this company
            filtered_df = filtered_df[filtered_df['unique_id'].str.startswith(f"{selected_company}_")]
        elif selected_program:
            # Only program selected: show all companies for this program
            filtered_df = filtered_df[filtered_df['unique_id'].str.endswith(f"_{selected_program}")]
        # If both are "All" (neither selected), show all company-program combinations (no filtering)
        
    elif selected_level == "State + Program":
        if selected_state and selected_program:
            # Both selected: unique_id is "program_state" (e.g., "clinical_leave_evaluation_CA")
            target_id = f"{selected_program}_{selected_state}"
            filtered_df = filtered_df[filtered_df['unique_id'] == target_id]
        elif selected_program:
            # Only program selected: show all states for this program
            filtered_df = filtered_df[filtered_df['unique_id'].str.startswith(f"{selected_program}_")]
        elif selected_state:
            # Only state selected: show all programs for this state
            filtered_df = filtered_df[filtered_df['unique_id'].str.endswith(f"_{selected_state}")]
        # If both are "All" (neither selected), show all program-state combinations (no filtering)
        
    elif selected_level == "Company + State + Program":
        if selected_company and selected_state and selected_program:
            # All three selected: unique_id is "company_program_state" (e.g., "acorabloomcorp_clinical_leave_evaluation_CA")
            target_id = f"{selected_company}_{selected_program}_{selected_state}"
            filtered_df = filtered_df[filtered_df['unique_id'] == target_id]
        else:
            # For partial selections, use string matching
            filters = []
            if selected_company:
                filters.append(filtered_df['unique_id'].str.startswith(f"{selected_company}_"))
            if selected_program:
                filters.append(filtered_df['unique_id'].str.contains(f"_{selected_program}_"))
            if selected_state:
                filters.append(filtered_df['unique_id'].str.endswith(f"_{selected_state}"))
            
            if filters:
                # Combine all filters with AND logic
                combined_filter = filters[0]
                for f in filters[1:]:
                    combined_filter = combined_filter & f
                filtered_df = filtered_df[combined_filter]
        # If all are "All" (nothing selected), show all company-program-state combinations (no filtering)
    
    return filtered_df

def extract_metrics_from_original(eval_df):
    """Extract metrics from original evaluation files"""
    models = ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML']
    
    if eval_df.empty:
        return pd.DataFrame()
    
    if len(eval_df) == 1:
        # Single entry - show exact metrics
        row = eval_df.iloc[0]
        metrics_data = []
        for model in models:
            metrics_data.append({
                'Model': model,
                'MAE': round(row[f'MAE_{model}'], 2),
                'RMSE': round(row[f'RMSE_{model}'], 2),
                'MAPE (%)': round(row[f'MAPE_{model}'], 2),
                'WMAPE (%)': round(row[f'WMAPE_{model}'], 2)
            })
        return pd.DataFrame(metrics_data)
    else:
        # Multiple entries - aggregate (mean for MAE/RMSE, median for MAPE/WMAPE)
        metrics_data = []
        for model in models:
            metrics_data.append({
                'Model': model,
                'MAE': round(eval_df[f'MAE_{model}'].mean(), 2),
                'RMSE': round(eval_df[f'RMSE_{model}'].mean(), 2),
                'MAPE (%)': round(eval_df[f'MAPE_{model}'].median(), 2),
                'WMAPE (%)': round(eval_df[f'WMAPE_{model}'].median(), 2)
            })
        return pd.DataFrame(metrics_data)

def load_original_summary(split_date, horizon, frequency='weekly'):
    """Load overall summary from original evaluation files"""
    summary_tables = {}
    eval_dir = f"data/evaluation_{split_date}_{horizon}"
    
    # Load pre-computed summaries (these are already correct)
    for metric in ['MAE', 'RMSE', 'MAPE', 'WMAPE']:
        # Try new naming pattern first (2024-01-01 onwards)
        file_path = f"{eval_dir}/{frequency}_{metric.lower()}_summary.parquet"
        if os.path.exists(file_path):
            summary_tables[metric] = pd.read_parquet(file_path)
        else:
            # Try old naming pattern (2024-09-30 and earlier)
            file_path = f"{eval_dir}/{frequency}_summary_{metric}.parquet"
            if os.path.exists(file_path):
                summary_tables[metric] = pd.read_parquet(file_path)
    
    # For MAPE and WMAPE, compute median from individual evaluations if not pre-computed
    for metric in ['MAPE', 'WMAPE']:
        if metric in summary_tables:
            continue  # Skip if already loaded from pre-computed file
        level_data = {}
        models = ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML']
        
        level_mapping = {
            'overall': 'overall',
            'company': 'company', 
            'state': 'state',
            'program': 'program',
            'company_state': 'company_state',
            'company_program': 'company_program',
            'program_state': 'program_state',
            'company_program_state': 'company_program_state'
        }
        
        for level_name, file_suffix in level_mapping.items():
            file_path = f"{eval_dir}/{frequency}_{file_suffix}_evaluation.parquet"
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                level_medians = []
                
                for model in models:
                    metric_col = f'{metric}_{model}'
                    if metric_col in df.columns:
                        median_val = df[metric_col].median()
                        level_medians.append(median_val)
                    else:
                        level_medians.append(0.0)
                
                level_data[level_name] = level_medians
        
        if level_data:
            summary_tables[metric] = pd.DataFrame(level_data, index=models).round(2)
    
    return summary_tables

def get_aggregated_data(forecast_data, level, company=None, state=None, program=None):
    """Get aggregated data based on hierarchy level"""
    
    if level == "Overall":
        # Aggregate all data directly
        agg_data = forecast_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
        agg_data['unique_id'] = 'Overall'
        return agg_data
    
    elif level == "Company":
        if company:
            # Specific company - aggregate across states and programs for this company
            company_data = forecast_data[forecast_data['company'] == company]
            agg_data = company_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"Company_{company}"
            return agg_data
        else:
            # All companies - take the first company by default to show company-level view
            companies = sorted(forecast_data['company'].unique())
            first_company = companies[0] if companies else None
            if first_company:
                company_data = forecast_data[forecast_data['company'] == first_company]
                agg_data = company_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
                agg_data['unique_id'] = f"Company_Level_Sample_{first_company}"
                return agg_data
            else:
                return pd.DataFrame()
    
    elif level == "State":
        if state:
            # Specific state - aggregate across companies and programs for this state
            state_data = forecast_data[forecast_data['state'] == state]
            agg_data = state_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"State_{state}"
            return agg_data
        else:
            # All states - take the first state by default to show state-level view
            states = sorted(forecast_data['state'].unique())
            first_state = states[0] if states else None
            if first_state:
                state_data = forecast_data[forecast_data['state'] == first_state]
                agg_data = state_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
                agg_data['unique_id'] = f"State_Level_Sample_{first_state}"
                return agg_data
            else:
                return pd.DataFrame()
    
    elif level == "Program":
        if program:
            # Specific program - aggregate across companies and states for this program
            program_data = forecast_data[forecast_data['program'] == program]
            agg_data = program_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"Program_{program}"
            return agg_data
        else:
            # All programs - take the first program by default to show program-level view
            programs = sorted(forecast_data['program'].unique())
            first_program = programs[0] if programs else None
            if first_program:
                program_data = forecast_data[forecast_data['program'] == first_program]
                agg_data = program_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
                agg_data['unique_id'] = f"Program_Level_Sample_{first_program}"
                return agg_data
            else:
                return pd.DataFrame()
    
    elif level == "Company + State":
        if company and state:
            # Specific company + state
            filtered_data = forecast_data[
                (forecast_data['company'] == company) & 
                (forecast_data['state'] == state)
            ]
            agg_data = filtered_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"CompanyState_{company}_{state}"
            return agg_data
        elif company:
            # Specific company, aggregate across ALL states
            company_data = forecast_data[forecast_data['company'] == company]
            agg_data = company_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"Company_{company}_AllStates"
            return agg_data
        elif state:
            # Specific state, aggregate across ALL companies
            state_data = forecast_data[forecast_data['state'] == state]
            agg_data = state_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"AllCompanies_State_{state}"
            return agg_data
        else:
            # All companies + all states - find an existing combination for sample display
            existing_combinations = forecast_data[['company', 'state']].drop_duplicates()
            if len(existing_combinations) > 0:
                first_combo = existing_combinations.iloc[0]
                first_company = first_combo['company']
                first_state = first_combo['state']
                
                filtered_data = forecast_data[
                    (forecast_data['company'] == first_company) & 
                    (forecast_data['state'] == first_state)
                ]
                
                if len(filtered_data) > 0:
                    agg_data = filtered_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
                    agg_data['unique_id'] = f"CompanyState_Sample_{first_company}_{first_state}"
                    return agg_data
            
            return pd.DataFrame()

    elif level == "Company + Program":
        if company and program:
            # Specific company + program
            filtered_data = forecast_data[
                (forecast_data['company'] == company) & 
                (forecast_data['program'] == program)
            ]
            agg_data = filtered_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"CompanyProgram_{company}_{program}"
            return agg_data
        elif company:
            # Specific company, aggregate across ALL programs
            company_data = forecast_data[forecast_data['company'] == company]
            agg_data = company_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"Company_{company}_AllPrograms"
            return agg_data
        elif program:
            # Specific program, aggregate across ALL companies
            program_data = forecast_data[forecast_data['program'] == program]
            agg_data = program_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"AllCompanies_Program_{program}"
            return agg_data
        else:
            # All companies + all programs - find an existing combination for sample display
            existing_combinations = forecast_data[['company', 'program']].drop_duplicates()
            if len(existing_combinations) > 0:
                first_combo = existing_combinations.iloc[0]
                first_company = first_combo['company']
                first_program = first_combo['program']
                
                filtered_data = forecast_data[
                    (forecast_data['company'] == first_company) & 
                    (forecast_data['program'] == first_program)
                ]
                
                if len(filtered_data) > 0:
                    agg_data = filtered_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
                    agg_data['unique_id'] = f"CompanyProgram_Sample_{first_company}_{first_program}"
                    return agg_data
            
            return pd.DataFrame()

    elif level == "State + Program":
        if state and program:
            # Specific state + program
            filtered_data = forecast_data[
                (forecast_data['state'] == state) & 
                (forecast_data['program'] == program)
            ]
            agg_data = filtered_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"StateProgram_{state}_{program}"
            return agg_data
        elif state:
            # Specific state, aggregate across ALL programs
            state_data = forecast_data[forecast_data['state'] == state]
            agg_data = state_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"State_{state}_AllPrograms"
            return agg_data
        elif program:
            # Specific program, aggregate across ALL states
            program_data = forecast_data[forecast_data['program'] == program]
            agg_data = program_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"AllStates_Program_{program}"
            return agg_data
        else:
            # All states + all programs - find an existing combination for sample display
            existing_combinations = forecast_data[['state', 'program']].drop_duplicates()
            if len(existing_combinations) > 0:
                first_combo = existing_combinations.iloc[0]
                first_state = first_combo['state']
                first_program = first_combo['program']
                
                filtered_data = forecast_data[
                    (forecast_data['state'] == first_state) & 
                    (forecast_data['program'] == first_program)
                ]
                
                if len(filtered_data) > 0:
                    agg_data = filtered_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
                    agg_data['unique_id'] = f"StateProgram_Sample_{first_state}_{first_program}"
                    return agg_data
            
            return pd.DataFrame()
    
    elif level == "Company + State + Program":
        if company and state and program:
            # Specific combination - all three selected
            filtered_data = forecast_data[
                (forecast_data['company'] == company) & 
                (forecast_data['state'] == state) & 
                (forecast_data['program'] == program)
            ]
            if not filtered_data.empty:
                filtered_data = filtered_data.copy()
                filtered_data['unique_id'] = f"CompanyStateProgram_{company}_{state}_{program}"
                return filtered_data
        elif company and state:
            # Company + State selected, aggregate across ALL programs
            filtered_data = forecast_data[
                (forecast_data['company'] == company) & 
                (forecast_data['state'] == state)
            ]
            agg_data = filtered_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"CompanyState_{company}_{state}_AllPrograms"
            return agg_data
        elif company and program:
            # Company + Program selected, aggregate across ALL states
            filtered_data = forecast_data[
                (forecast_data['company'] == company) & 
                (forecast_data['program'] == program)
            ]
            agg_data = filtered_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"CompanyProgram_{company}_{program}_AllStates"
            return agg_data
        elif state and program:
            # State + Program selected, aggregate across ALL companies
            filtered_data = forecast_data[
                (forecast_data['state'] == state) & 
                (forecast_data['program'] == program)
            ]
            agg_data = filtered_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"AllCompanies_StateProgram_{state}_{program}"
            return agg_data
        elif company:
            # Only company selected, aggregate across ALL states and programs
            company_data = forecast_data[forecast_data['company'] == company]
            agg_data = company_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"Company_{company}_AllStatesPrograms"
            return agg_data
        elif state:
            # Only state selected, aggregate across ALL companies and programs
            state_data = forecast_data[forecast_data['state'] == state]
            agg_data = state_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"AllCompanies_State_{state}_AllPrograms"
            return agg_data
        elif program:
            # Only program selected, aggregate across ALL companies and states
            program_data = forecast_data[forecast_data['program'] == program]
            agg_data = program_data.groupby('ds')[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            agg_data['unique_id'] = f"AllCompanies_AllStates_Program_{program}"
            return agg_data
        else:
            # Nothing selected - find an existing combination for sample display
            existing_combinations = forecast_data[['company', 'state', 'program']].drop_duplicates()
            if len(existing_combinations) > 0:
                first_combo = existing_combinations.iloc[0]
                first_company = first_combo['company']
                first_state = first_combo['state']
                first_program = first_combo['program']
                
                filtered_data = forecast_data[
                    (forecast_data['company'] == first_company) & 
                    (forecast_data['state'] == first_state) & 
                    (forecast_data['program'] == first_program)
                ]
                
                if not filtered_data.empty:
                    filtered_data = filtered_data.copy()
                    filtered_data['unique_id'] = f"CompanyStateProgram_Sample_{first_company}_{first_state}_{first_program}"
                    return filtered_data
            
            return pd.DataFrame()
    
    return pd.DataFrame()



def aggregate_intervals_hierarchical(forecast_with_intervals, level, company=None, state=None, program=None, frequency='weekly'):
    """
    Aggregate prediction intervals up the hierarchy
    
    Parameters:
    -----------
    forecast_with_intervals : pd.DataFrame
        Bottom-level forecasts with intervals
    level : str
        Hierarchy level to aggregate to
    company, state, program : str, optional
        Specific selections for filtering
    frequency : str
        'weekly' or 'monthly' aggregation
        
    Returns:
    --------
    pd.DataFrame
        Aggregated forecasts with intervals
    """
    # Get available model columns
    model_cols = ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble']
    if 'ensemble_ML' in forecast_with_intervals.columns:
        model_cols.append('ensemble_ML')
    
    available_models = [col for col in model_cols if col in forecast_with_intervals.columns]
    
    # Apply time aggregation first if monthly
    df = forecast_with_intervals.copy()
    if frequency == 'monthly':
        df['period'] = df['ds'].dt.to_period('M')
        group_time = 'period'
    else:
        group_time = 'ds'
    
    # Define aggregation columns based on frequency
    if frequency == 'monthly':
        agg_cols = ['period']
    else:
        agg_cols = ['ds']
    
    # Apply filters based on level and selections
    if level == "Company" and company:
        df = df[df['company'] == company]
    elif level == "State" and state:
        df = df[df['state'] == state]
    elif level == "Program" and program:
        df = df[df['program'] == program]
    elif level == "Company + State":
        if company:
            df = df[df['company'] == company]
        if state:
            df = df[df['state'] == state]
    elif level == "Company + Program":
        if company:
            df = df[df['company'] == company]
        if program:
            df = df[df['program'] == program]
    elif level == "State + Program":
        if state:
            df = df[df['state'] == state]
        if program:
            df = df[df['program'] == program]
    elif level == "Company + State + Program":
        if company:
            df = df[df['company'] == company]
        if state:
            df = df[df['state'] == state]
        if program:
            df = df[df['program'] == program]
    
    if df.empty:
        return pd.DataFrame()
    
    # Aggregate: sum point forecasts, and combine intervals using quadrature
    agg_dict = {}
    
    # Point forecasts - simple sum
    for model in available_models:
        if model in df.columns:
            agg_dict[model] = 'sum'
    
    # Intervals - use quadrature rule: sqrt(sum(widths^2))
    for model in available_models:
        lo_col = f'{model}_lo'
        hi_col = f'{model}_hi'
        if lo_col in df.columns and hi_col in df.columns:
            # Compute interval widths first
            df[f'{model}_width'] = (df[hi_col] - df[lo_col]) / 2
    
    # Group and aggregate
    if agg_cols:
        grouped = df.groupby(agg_cols)
        
        # Aggregate point forecasts
        agg_data = grouped[available_models].sum().reset_index()
        
        # Aggregate interval widths using quadrature
        for model in available_models:
            width_col = f'{model}_width'
            if width_col in df.columns:
                # Quadrature aggregation: sqrt(sum(width^2))
                agg_widths = grouped[width_col].apply(lambda x: np.sqrt((x**2).sum())).reset_index()
                agg_data[f'{model}_agg_width'] = agg_widths[width_col]
                
                # Reconstruct intervals
                agg_data[f'{model}_lo'] = np.maximum(0.0, agg_data[model] - agg_data[f'{model}_agg_width'])
                agg_data[f'{model}_hi'] = agg_data[model] + agg_data[f'{model}_agg_width']
                
                # Clean up
                agg_data = agg_data.drop(columns=[f'{model}_agg_width'])
        
        # Convert period back to datetime if needed
        if frequency == 'monthly' and 'period' in agg_data.columns:
            agg_data['ds'] = agg_data['period'].dt.start_time
            agg_data = agg_data.drop(columns=['period'])
        
        return agg_data
    
    return pd.DataFrame()

# Set page config
st.set_page_config(
    page_title="Forecasting Dashboard",
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .forecast-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .summary-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 2rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="forecast-header">
    <h1>📈  Forecasting Dashboard</h1>
    <p>Hierarchical Time Series Forecasting with Multi-Level Validation</p>
</div>
""", unsafe_allow_html=True)

def plot_forecast_single(series_data, series_name, show_intervals=True):
    """Plot forecast for a single series with optional prediction intervals"""
    if series_data.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    fig = go.Figure()
    
    models = ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML']
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'pink']
    
    series_data = series_data.sort_values('ds')
    
    # Plot actual values
    if 'y' in series_data.columns:
        fig.add_trace(
            go.Scatter(
                x=series_data['ds'],
                y=series_data['y'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='black', width=3),
                marker=dict(size=6)
            )
        )
    
    # Plot forecasts with intervals
    for j, model in enumerate(models):
        if model in series_data.columns:
            # Plot confidence interval first (so it appears behind the line)
            if show_intervals and f'{model}_lo' in series_data.columns and f'{model}_hi' in series_data.columns:
                # Create confidence band
                fig.add_trace(
                    go.Scatter(
                        x=list(series_data['ds']) + list(series_data['ds'][::-1]),
                        y=list(series_data[f'{model}_hi']) + list(series_data[f'{model}_lo'][::-1]),
                        fill='toself',
                        fillcolor=f'rgba({",".join(str(int(x*255)) for x in px.colors.hex_to_rgb(colors[j]))}, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        name=f'{model} 90% CI'
                    )
                )
            
            # Plot point forecast
            fig.add_trace(
                go.Scatter(
                    x=series_data['ds'],
                    y=series_data[model],
                    mode='lines',
                    name=f'{model}',
                    line=dict(color=colors[j], width=3 if model == 'ensemble' else 2, 
                             dash='solid' if model == 'ensemble' else 'dash')
                )
            )
    
    fig.update_layout(
        title=f"Forecast for {series_name}",
        xaxis_title="Date",
        yaxis_title="Sessions",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def calculate_metrics_for_series(series_data):
    """Calculate metrics for a single series or aggregated data"""
    models = ['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML']
    metrics_data = []
    
    if series_data.empty or 'y' not in series_data.columns:
        return pd.DataFrame()
    
    # Handle aggregated data (multiple rows with same unique_id)
    if len(series_data) > 1 and 'unique_id' in series_data.columns:
        # Group by unique_id to handle any potential duplicates
        agg_series = series_data.groupby('unique_id').first().reset_index()
        if len(agg_series) == 1:
            series_data = series_data.copy()
    
    # Remove any NaN or infinite values
    series_data = series_data.dropna(subset=['y'])
    
    # Get actual values
    y = series_data['y'].values
    
    # Filter out any remaining invalid values
    valid_mask = np.isfinite(y) & (y >= 0)
    y = y[valid_mask]
    
    if len(y) == 0:
        return pd.DataFrame()
    
    for model in models:
        if model in series_data.columns:
            yhat = series_data[model].values[valid_mask]
            
            # Ensure same length
            if len(yhat) != len(y):
                continue
            
            # Filter out invalid predictions
            pred_valid_mask = np.isfinite(yhat) & (yhat >= 0)
            y_clean = y[pred_valid_mask]
            yhat_clean = yhat[pred_valid_mask]
            
            if len(y_clean) == 0:
                continue
            
            # Calculate metrics
            mae = np.mean(np.abs(y_clean - yhat_clean))
            mse = np.mean((y_clean - yhat_clean) ** 2)
            rmse = np.sqrt(mse)
            
            # MAPE (skip zeros)
            mask = y_clean != 0
            if mask.any():
                mape = np.mean(np.abs((y_clean[mask] - yhat_clean[mask]) / y_clean[mask])) * 100
            else:
                mape = 0.0
            
            # WMAPE
            denom = np.sum(np.abs(y_clean))
            wmape = (np.sum(np.abs(y_clean - yhat_clean)) / denom * 100) if denom > 0 else 0.0
            
            metrics_data.append({
                'Model': model,
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'MAPE (%)': round(mape, 2),
                'WMAPE (%)': round(wmape, 2)
            })
    
    return pd.DataFrame(metrics_data)

# Sidebar for navigation and controls
with st.sidebar:
    st.header("🔧 Control Panel")
    
    # Main mode selection
    app_mode = st.selectbox(
        "Select Mode:",
        ["🔍 Validation", "🚀 Forecasting"],
        index=0
    )
    
    st.divider()
    
    if app_mode == "🔍 Validation":
        st.subheader("📊 Validation Analysis")
        
        # Check for existing forecast files
        forecast_files = glob.glob('data/all_bottom_forecast_*.parquet')
        
        if forecast_files:
            st.markdown("**Available Results:**")
            
            # Extract available results
            available_results = []
            
            for file in forecast_files:
                basename = os.path.basename(file)
                parts = basename.replace('all_bottom_forecast_', '').replace('.parquet', '').split('_')
                if len(parts) >= 2:
                    date_part = '_'.join(parts[:-1])
                    horizon = parts[-1]
                    
                    # Load the file to get actual date range
                    try:
                        forecast_data = pd.read_parquet(file)
                        forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
                        start_date = forecast_data['ds'].min().strftime('%Y-%m-%d')
                        end_date = forecast_data['ds'].max().strftime('%Y-%m-%d')
                        display_text = f"📅 {start_date} to {end_date}"
                    except Exception:
                        # Fallback to original format if file can't be read
                        display_text = f"📅 {date_part} ({horizon}w)"
                    
                    available_results.append({
                        'display': display_text,
                        'split_date': date_part,
                        'horizon': horizon,
                        'forecast_file': file
                    })
            
            if available_results:
                result_options = [r['display'] for r in available_results]
                selected_result_display = st.selectbox("Choose Results:", result_options)
                
                # Find selected result
                selected_result = None
                for r in available_results:
                    if r['display'] == selected_result_display:
                        selected_result = r
                        break
                
                if st.button("🔥 Load Results", type="primary", width="stretch"):
                    with st.spinner("Loading forecast data..."):
                        try:
                            # Load forecast data
                            forecast_data = pd.read_parquet(selected_result['forecast_file'])
                            st.session_state.forecast_data = forecast_data
                            st.session_state.split_date = selected_result['split_date']
                            st.session_state.horizon = selected_result['horizon']
                            
                            st.success(f"✅ Loaded: {os.path.basename(selected_result['forecast_file'])}")
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Error loading file: {str(e)}")
            else:
                st.warning("📂 No valid forecast files found")
        else:
            st.warning("📂 No forecast files found")

# Main content area
if app_mode == "🔍 Validation":
    st.header("🔍 Model Validation Results")
    
    # Display results if available
    if 'forecast_data' in st.session_state:
        
        forecast_data = st.session_state.forecast_data
        
        # Hierarchy level selector
        st.subheader("🏗️ Select Hierarchy Level")
        level_options = [
            "Overall",
            "Company", 
            "State",
            "Program",
            "Company + State",
            "Company + Program", 
            "State + Program",
            "Company + State + Program"
        ]
        
        selected_level = st.selectbox("Choose aggregation level:", level_options, index=0)
        
        # Initialize selections
        selected_company = None
        selected_state = None  
        selected_program = None
        
        # Create dynamic selectors based on level - cleaner logic
        col1, col2, col3 = st.columns(3)
        
        # Company selector logic
        with col1:
            show_company_selector = (
                selected_level == "Company" or 
                "Company" in selected_level
            )
            
            if show_company_selector and 'company' in forecast_data.columns:
                companies = ['All'] + sorted(forecast_data['company'].unique().tolist())
                selected_company = st.selectbox(
                    "🏢 Company:", 
                    companies, 
                    index=0, 
                    key=f"company_{selected_level}"
                )
                if selected_company == 'All':
                    selected_company = None
        
        # State selector logic
        with col2:
            show_state_selector = (
                selected_level == "State" or 
                "State" in selected_level
            )
            
            if show_state_selector and 'state' in forecast_data.columns:
                # Filter states based on company selection if applicable
                if selected_company and "Company" in selected_level:
                    available_states = forecast_data[
                        forecast_data['company'] == selected_company
                    ]['state'].unique()
                else:
                    available_states = forecast_data['state'].unique()
                
                states = ['All'] + sorted(available_states.tolist())
                selected_state = st.selectbox(
                    "🗺️ State:", 
                    states, 
                    index=0, 
                    key=f"state_{selected_level}"
                )
                if selected_state == 'All':
                    selected_state = None
            else:
                st.empty()
        
        # Program selector logic
        with col3:
            show_program_selector = (
                selected_level == "Program" or 
                "Program" in selected_level
            )
            
            if show_program_selector and 'program' in forecast_data.columns:
                # Filter programs based on company and state selections if applicable
                filtered_data = forecast_data.copy()
                
                if selected_company and "Company" in selected_level:
                    filtered_data = filtered_data[filtered_data['company'] == selected_company]
                
                if selected_state and "State" in selected_level:
                    filtered_data = filtered_data[filtered_data['state'] == selected_state]
                
                available_programs = filtered_data['program'].unique()
                programs = ['All'] + sorted(available_programs.tolist())
                
                selected_program = st.selectbox(
                    "📋 Program:", 
                    programs, 
                    index=0, 
                    key=f"program_{selected_level}"
                )
                if selected_program == 'All':
                    selected_program = None
            else:
                st.empty()
        
        # Add weekly/monthly selector for individual series
        st.subheader("⚙️ Validation Settings")
        validation_cols = st.columns(2)
        with validation_cols[0]:
            series_frequency = st.radio(
                "📅 Series View:",
                ["Weekly", "Monthly"],
                horizontal=True,
                key="series_freq"
            )

        
        # Check if we should show data for combination levels
        should_show_data = True
        show_aggregated_level_metrics = False
        
        if selected_level in ["Company", "State", "Program"]:
            # For single levels, determine what to show
            if selected_level == "Company":
                if not selected_company:
                    # All companies - show aggregated level metrics
                    should_show_data = True
                    show_aggregated_level_metrics = True
                else:
                    # Specific company selected - show specific combination
                    should_show_data = True
                    show_aggregated_level_metrics = False
            elif selected_level == "State":
                if not selected_state:
                    # All states - show aggregated level metrics
                    should_show_data = True
                    show_aggregated_level_metrics = True
                else:
                    # Specific state selected - show specific combination
                    should_show_data = True
                    show_aggregated_level_metrics = False
            elif selected_level == "Program":
                if not selected_program:
                    # All programs - show aggregated level metrics
                    should_show_data = True
                    show_aggregated_level_metrics = True
                else:
                    # Specific program selected - show specific combination
                    should_show_data = True
                    show_aggregated_level_metrics = False
        elif selected_level in ["Company + State", "Company + Program", "State + Program", "Company + State + Program"]:
            # For combination levels, determine what to show
            if selected_level == "Company + State":
                if not selected_company and not selected_state:
                    # All/All - show aggregated level metrics
                    should_show_data = True
                    show_aggregated_level_metrics = True
                elif selected_company and selected_state:
                    # Both selected - show specific combination
                    should_show_data = True
                    show_aggregated_level_metrics = False
                else:
                    # Partial selection - show nothing
                    should_show_data = False
            elif selected_level == "Company + Program":
                if not selected_company and not selected_program:
                    # All/All - show aggregated level metrics
                    should_show_data = True
                    show_aggregated_level_metrics = True
                elif selected_company and selected_program:
                    # Both selected - show specific combination
                    should_show_data = True
                    show_aggregated_level_metrics = False
                else:
                    # Partial selection - show nothing
                    should_show_data = False
            elif selected_level == "State + Program":
                if not selected_state and not selected_program:
                    # All/All - show aggregated level metrics
                    should_show_data = True
                    show_aggregated_level_metrics = True
                elif selected_state and selected_program:
                    # Both selected - show specific combination
                    should_show_data = True
                    show_aggregated_level_metrics = False
                else:
                    # Partial selection - show nothing
                    should_show_data = False
            elif selected_level == "Company + State + Program":
                if not selected_company and not selected_state and not selected_program:
                    # All/All/All - show aggregated level metrics
                    should_show_data = True
                    show_aggregated_level_metrics = True
                elif selected_company and selected_state and selected_program:
                    # All three selected - show specific combination
                    should_show_data = True
                    show_aggregated_level_metrics = False
                else:
                    # Partial selection - show nothing
                    should_show_data = False
        
        if should_show_data:
            # Get aggregated data only if not showing level metrics
            if not show_aggregated_level_metrics:
                agg_data = get_aggregated_data(forecast_data, selected_level, selected_company, selected_state, selected_program)
                
                # Apply monthly aggregation if selected
                if series_frequency == "Monthly" and not agg_data.empty:
                    agg_data = agg_data.groupby(
                        ['unique_id', pd.Grouper(key='ds', freq='ME')]
                    )[['TimeGPT', 'LGB', 'XGB', 'RF', 'ensemble', 'ensemble_ML', 'y']].sum().reset_index()
            else:
                agg_data = pd.DataFrame()  # No plot data for aggregated level metrics
            
            # Create two columns for plot and metrics
            plot_col, metrics_col = st.columns([2, 1])
            
            with plot_col:
                st.subheader("📈 Forecast Plot")
                
                if show_aggregated_level_metrics:
                    # Show message for aggregated level view
                    st.info("📊 **Select specific combination to see forecast plot**")
                    st.markdown("Choose specific company, state, and/or program from the selectors above to view the detailed forecast visualization.")
                    
                    # Show what level metrics are being displayed
                    level_description = {
                        "Company + State": "Showing aggregated metrics across all company-state combinations",
                        "Company + Program": "Showing aggregated metrics across all company-program combinations",
                        "State + Program": "Showing aggregated metrics across all program-state combinations",
                        "Company + State + Program": "Showing aggregated metrics across all company-program-state combinations"
                    }
                    
                    if selected_level in level_description:
                        st.caption(level_description[selected_level])
                elif not agg_data.empty:
                    # Show forecast plot for specific combination
                    series_name = get_series_display_name(selected_level, selected_company, selected_state, selected_program, forecast_data)
                    series_name += f" ({series_frequency})"
                    fig = plot_forecast_single(agg_data, series_name)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.warning("No forecast data available for this combination")
            
            with metrics_col:
                st.subheader("📊 Performance Metrics")
                
                # Set defaults if not in session state
                split_date = st.session_state.get('split_date', '2024-09-30')
                horizon = st.session_state.get('horizon', '13')
                
                frequency_key = series_frequency.lower()
                
                # Load the evaluation file for the selected level
                eval_df = load_original_evaluation(
                    split_date,
                    horizon, 
                    selected_level,
                    frequency_key
                )
                
                if not eval_df.empty:
                    if show_aggregated_level_metrics:
                        # Show aggregated metrics for the entire level (All/All case)
                        metrics_df = extract_metrics_from_original(eval_df)
                        
                        if not metrics_df.empty:
                            st.dataframe(
                                metrics_df,
                                width="stretch",
                                hide_index=True
                            )
                            
                            # Show champion and challenger models
                            champion, challenger = get_champion_challenger(metrics_df)
                            if champion:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("### 🏆 Champion")
                                    st.metric(
                                        label="Lowest RMSE",
                                        value=champion['model'],
                                        delta=f"RMSE: {champion['rmse']}"
                                    )
                                if challenger:
                                    with col2:
                                        st.markdown("### 🥈 Challenger")
                                        st.metric(
                                            label="Second Lowest RMSE",
                                            value=challenger['model'],
                                            delta=f"RMSE: {challenger['rmse']}"
                                        )
                            else:
                                st.error("❌ Could not extract metrics")
                        else:
                            st.error("❌ Could not extract metrics")
                    else:
                        # Show metrics for the specific combination
                        filtered_eval = filter_original_evaluation(
                            eval_df, selected_level, selected_company, selected_state, selected_program
                        )
                        
                        if not filtered_eval.empty:
                            metrics_df = extract_metrics_from_original(filtered_eval)
                            
                            if not metrics_df.empty:
                                st.dataframe(
                                    metrics_df,
                                    width="stretch",
                                    hide_index=True
                                )
                                
                                # Show champion and challenger models
                                champion, challenger = get_champion_challenger(metrics_df)
                                if champion:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("### 🏆 Champion")
                                        st.metric(
                                            label="Lowest RMSE",
                                            value=champion['model'],
                                            delta=f"RMSE: {champion['rmse']}"
                                        )
                                    if challenger:
                                        with col2:
                                            st.markdown("### 🥈 Challenger")
                                            st.metric(
                                                label="Second Lowest RMSE",
                                                value=challenger['model'],
                                                delta=f"RMSE: {challenger['rmse']}"
                                            )
                            else:
                                st.error("❌ Could not extract metrics")
                        else:
                            st.warning("⚠️ No data found for selected combination")
                else:
                    st.error("❌ No evaluation data found for this configuration")
        else:
            # For combination levels when partial selections are made, show instruction message
            st.info("📊 **Complete the combination to see forecast and performance metrics**")
            
            # Show what selections are needed
            if selected_level == "Company + State":
                st.markdown("**Required:** Select both Company AND State (or leave both as 'All' for aggregated view)")
            elif selected_level == "Company + Program":
                st.markdown("**Required:** Select both Company AND Program (or leave both as 'All' for aggregated view)")
            elif selected_level == "State + Program":
                st.markdown("**Required:** Select both State AND Program (or leave both as 'All' for aggregated view)")
            elif selected_level == "Company + State + Program":
                st.markdown("**Required:** Select all three: Company AND State AND Program (or leave all as 'All' for aggregated view)")
        
        # Overall Summary Section - Load Pre-computed Tables
        st.subheader("📊 Overall Summary - All Hierarchy Levels")
        
        summary_cols = st.columns([1, 1])
        
        with summary_cols[0]:
            summary_frequency = st.radio(
                "📅 Frequency:",
                ["Weekly", "Monthly"],
                horizontal=True,
                key="summary_freq"
            )
        
        with summary_cols[1]:
            summary_metric = st.selectbox(
                "📏 Metric:",
                ['RMSE', 'MAE', 'MAPE', 'WMAPE'],
                index=0,
                key="summary_metric"
            )
        
        # Load pre-computed summary tables
        # Set defaults if not in session state
        split_date = st.session_state.get('split_date', '2024-09-30')
        horizon = st.session_state.get('horizon', '13')
        
        frequency = summary_frequency.lower()
        summary_tables = load_original_summary(
            split_date, 
            horizon, 
            frequency
        )
            
        if summary_tables and summary_metric in summary_tables:
            # Display summary table for selected metric
            st.markdown(f"### {summary_frequency} {summary_metric} - Models × Hierarchy Levels")
            summary_df = summary_tables[summary_metric]
            st.dataframe(
                summary_df, 
                width="stretch"
            )
        else:
            st.warning(f"⚠️ No pre-computed summary tables found for {frequency}")
        
        # Feature Importance Section
        st.markdown("---")
        st.subheader("🔍 Feature Importance Analysis")
        
        # Load feature importance data
        feature_importance_data = load_feature_importance(split_date, horizon)
        
        if feature_importance_data:
            # Show 3 clean charts side by side
            st.markdown("### 📊 Feature Importance by Model")
            
            model_cols = st.columns(3)
            models = ['LGB', 'XGB', 'RF']
            
            for i, model in enumerate(models):
                with model_cols[i]:
                    if model in feature_importance_data:
                        st.markdown(f"**{model} Model**")
                        
                        # Create dataframe for this model
                        importance_df = pd.DataFrame([
                            {'Feature': feature, 'Importance': importance}
                            for feature, importance in feature_importance_data[model].items()
                        ]).sort_values('Importance', ascending=True)  # For horizontal bar chart
                        
                        # Create horizontal bar chart
                        fig = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title=f'{model}',
                            height=400
                        )
                        fig.update_layout(
                            showlegend=False,
                            xaxis=dict(range=[0, 1]),
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        fig.update_traces(
                            texttemplate='%{x:.3f}',
                            textposition='outside'
                        )
                        
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.info(f"{model} data not available")
            
            # All features list by type
            st.markdown("---")
            st.markdown("### 🔍 Complete Feature List by Type")
            
            # Define all features based on the pipeline
            feature_types = {
                "📈 Lag Features": [
                    "lag1", "lag2", "lag3", "lag4", "lag8", "lag12", "lag26", "lag52"
                ],
                "📊 Rolling Statistics": [
                    "rolling_mean_lag1_window_size4", "rolling_mean_lag1_window_size8", 
                    "rolling_mean_lag1_window_size12", "rolling_std_lag1_window_size4", 
                    "rolling_std_lag1_window_size8", "rolling_mean_lag2_window_size4", 
                    "rolling_mean_lag2_window_size8", "rolling_std_lag4_window_size4", 
                    "rolling_mean_lag8_window_size4", "rolling_mean_lag26_window_size4", 
                    "rolling_mean_lag52_window_size4"
                ],
                "🌊 Seasonal & Exponentially Weighted": [
                    "exponentially_weighted_mean_lag1_alpha0.3",
                    "seasonal_rolling_mean_lag52_season_length52_window_size4"
                ],
                "📅 Date Features": [
                    "week", "month", "quarter"
                ],
                "🏢 Categorical Features (Encoded)": [
                    "company_enc", "program_enc", "state_enc", 
                    "configuration_enc", "final_industry_group_enc"
                ],
                "📊 Customer Metrics": [
                    "member_count", "subscriber_count", "non_subscriber_count",
                    "subscriber_to_member_ratio", "cancellation_limit", "numeric_of_sessions"
                ]
            }
            
            # Display features by type in columns
            cols = st.columns(2)
            
            categories = list(feature_types.keys())
            for i, (category, features) in enumerate(feature_types.items()):
                with cols[i % 2]:
                    st.markdown(f"#### {category}")
                    st.markdown(f"*{len(features)} features*")
                    
                    for feature in features:
                        # Count how many models contain this feature
                        model_count = 0
                        models_with_feature = ['LGB', 'XGB', 'RF']  # The 3 tree-based models
                        
                        for model in models_with_feature:
                            if (model in feature_importance_data and 
                                isinstance(feature_importance_data[model], dict) and 
                                feature in feature_importance_data[model]):
                                model_count += 1
                        
                        # Display stars based on how many models contain the feature
                        if model_count == 3:
                            st.markdown(f"• **{feature}** ⭐⭐⭐")
                        elif model_count == 2:
                            st.markdown(f"• **{feature}** ⭐⭐")
                        elif model_count == 1:
                            st.markdown(f"• **{feature}** ⭐")
                        else:
                            st.markdown(f"• {feature}")
                    
                    st.markdown("")  # spacing
                
        else:
            st.info("Feature importance data not found for this forecast period.")
            

else:  # Forecasting mode
    st.header("📈 Forecast Analysis")
    
    # Check for latest forecast file specifically
    latest_forecast_file = 'data/all_bottom_forecast_latest.parquet'
    
    if os.path.exists(latest_forecast_file):
        # Load the latest forecast data
        try:
            df = pd.read_parquet(latest_forecast_file)
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Get file info
            forecast_weeks = df['ds'].nunique()
            first_date = df['ds'].min()
            last_date = df['ds'].max()
            
            # Show forecast info
            st.info(f"📅 **Forecast Period:** {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')} ({forecast_weeks} weeks)")
            
            # Use intervals if available in the file
            df_with_intervals = df
            
            # Display forecast dashboard
            st.subheader("📊 Forecast Dashboard")
            
            # Hierarchical analysis
            st.subheader("🔍 Hierarchical Forecast Analysis")
            
            # Analysis controls
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            
            with analysis_col1:
                # Level selector
                level_options = [
                    "Overall", "Company", "Program", "State", 
                    "Company + Program", "Company + State", "Program + State", "Company + Program + State"
                ]
                selected_level = st.selectbox("📊 Aggregation Level:", level_options)
            
            with analysis_col2:
                # Forecast horizon selector (number of weeks)
                max_weeks = df['ds'].nunique()
                
                # Create horizon options
                horizon_options = ["All Weeks"]
                
                # Add common horizon periods
                common_horizons = [4, 8, 13, 26, 52]
                for horizon in common_horizons:
                    if horizon <= max_weeks:
                        horizon_options.append(f"{horizon} weeks")
                
                # Add max available if not already included
                if max_weeks not in common_horizons and max_weeks > 1:
                    horizon_options.append(f"{max_weeks} weeks (Full)")
                
                selected_horizon = st.selectbox("📏 Forecast Horizon:", horizon_options)
            
            with analysis_col3:
                # Weekly vs Monthly aggregation
                aggregation_type = st.selectbox(
                    "📅 Time Aggregation:", 
                    ["Weekly", "Monthly"],
                    help="Aggregate forecasts by week or month"
                )
            

            
            # Filter controls based on level
            companies = sorted(df['company'].unique())
            programs = sorted(df['program'].unique())
            states = sorted(df['state'].unique())
            
            # Dynamic filters
            filters_col1, filters_col2, filters_col3 = st.columns(3)
            
            with filters_col1:
                if selected_level in ["Company", "Company + Program", "Company + State", "Company + Program + State"]:
                    selected_company = st.selectbox("🏢 Company:", ["All"] + companies, key="forecast_company_filter")
                else:
                    selected_company = None
            
            with filters_col2:
                if selected_level in ["Program", "Company + Program", "Program + State", "Company + Program + State"]:
                    selected_program = st.selectbox("📋 Program:", ["All"] + programs, key="forecast_program_filter")
                else:
                    selected_program = None
            
            with filters_col3:
                if selected_level in ["State", "Company + State", "Program + State", "Company + Program + State"]:
                    selected_state = st.selectbox("🗺️ State:", ["All"] + states, key="forecast_state_filter")
                else:
                    selected_state = None
            
            # Aggregate data based on level and filters using interval-aware aggregation
            display_df = df_with_intervals.copy()
            
            # Apply horizon filter
            if selected_horizon != "All Weeks":
                # Extract number of weeks from selection
                horizon_weeks = int(selected_horizon.split(" weeks")[0])
                
                # Take only the first N weeks
                all_dates = sorted(display_df['ds'].unique())
                if len(all_dates) > horizon_weeks:
                    cutoff_date = all_dates[horizon_weeks - 1]
                    display_df = display_df[display_df['ds'] <= cutoff_date]
            
            # Check if we should show data based on level and selections (same logic as validation)
            should_show_data = False
            show_aggregated_level_metrics = False
            
            # Use hierarchical aggregation with intervals
            freq_mapping = {"Weekly": "weekly", "Monthly": "monthly"}
            frequency = freq_mapping[aggregation_type]
            
            # Determine company, state, program selections
            comp_sel = selected_company if selected_company != "All" else None
            state_sel = selected_state if selected_state != "All" else None
            prog_sel = selected_program if selected_program != "All" else None
            
            if selected_level == "Overall":
                should_show_data = True
                show_aggregated_level_metrics = True
                agg_df = aggregate_intervals_hierarchical(
                    display_df, "Overall", 
                    company=None, state=None, program=None, 
                    frequency=frequency
                )
                title_suffix = f"Overall ({aggregation_type})"
                
            elif selected_level in ["Company", "Program", "State"]:
                # Single levels - always show metrics
                should_show_data = True
                show_aggregated_level_metrics = True
                agg_df = aggregate_intervals_hierarchical(
                    display_df, selected_level, 
                    company=comp_sel, state=state_sel, program=prog_sel, 
                    frequency=frequency
                )
                
                if selected_level == "Company":
                    title_suffix = f"Company: {selected_company} ({aggregation_type})" if selected_company != "All" else f"All Companies ({aggregation_type})"
                elif selected_level == "Program": 
                    title_suffix = f"Program: {selected_program} ({aggregation_type})" if selected_program != "All" else f"All Programs ({aggregation_type})"
                else:  # State
                    title_suffix = f"State: {selected_state} ({aggregation_type})" if selected_state != "All" else f"All States ({aggregation_type})"
                    
            else:
                # Combination levels - only show if all required selections are made
                if selected_level == "Company + Program":
                    if selected_company != "All" and selected_program != "All":
                        should_show_data = True
                        agg_df = aggregate_intervals_hierarchical(
                            display_df, selected_level, 
                            company=comp_sel, state=None, program=prog_sel, 
                            frequency=frequency
                        )
                        title_suffix = f"{selected_company} - {selected_program} ({aggregation_type})"
                    elif selected_company == "All" and selected_program == "All":
                        should_show_data = True
                        show_aggregated_level_metrics = True
                        agg_df = aggregate_intervals_hierarchical(
                            display_df, selected_level, 
                            company=None, state=None, program=None, 
                            frequency=frequency
                        )
                        title_suffix = f"All Company + Program Combinations ({aggregation_type})"
                        
                elif selected_level == "Company + State":
                    if selected_company != "All" and selected_state != "All":
                        should_show_data = True
                        agg_df = aggregate_intervals_hierarchical(
                            display_df, selected_level, 
                            company=comp_sel, state=state_sel, program=None, 
                            frequency=frequency
                        )
                        title_suffix = f"{selected_company} - {selected_state} ({aggregation_type})"
                    elif selected_company == "All" and selected_state == "All":
                        should_show_data = True
                        show_aggregated_level_metrics = True
                        agg_df = aggregate_intervals_hierarchical(
                            display_df, selected_level, 
                            company=None, state=None, program=None, 
                            frequency=frequency
                        )
                        title_suffix = f"All Company + State Combinations ({aggregation_type})"
                        
                elif selected_level == "Program + State":
                    if selected_program != "All" and selected_state != "All":
                        should_show_data = True
                        agg_df = aggregate_intervals_hierarchical(
                            display_df, selected_level, 
                            company=None, state=state_sel, program=prog_sel, 
                            frequency=frequency
                        )
                        title_suffix = f"{selected_program} - {selected_state} ({aggregation_type})"
                    elif selected_program == "All" and selected_state == "All":
                        should_show_data = True
                        show_aggregated_level_metrics = True
                        agg_df = aggregate_intervals_hierarchical(
                            display_df, selected_level, 
                            company=None, state=None, program=None, 
                            frequency=frequency
                        )
                        title_suffix = f"All Program + State Combinations ({aggregation_type})"
                        
                elif selected_level == "Company + Program + State":
                    if selected_company != "All" and selected_program != "All" and selected_state != "All":
                        should_show_data = True
                        agg_df = aggregate_intervals_hierarchical(
                            display_df, selected_level, 
                            company=comp_sel, state=state_sel, program=prog_sel, 
                            frequency=frequency
                        )
                        title_suffix = f"{selected_company} - {selected_program} - {selected_state} ({aggregation_type})"
                    elif selected_company == "All" and selected_program == "All" and selected_state == "All":
                        should_show_data = True
                        show_aggregated_level_metrics = True
                        agg_df = aggregate_intervals_hierarchical(
                            display_df, selected_level, 
                            company=None, state=None, program=None, 
                            frequency=frequency
                        )
                        title_suffix = f"All Company + Program + State Combinations ({aggregation_type})"
            
            if should_show_data and not agg_df.empty:
                # Add summary information prominently above the plot
                total_forecast = agg_df['ensemble'].sum()
                avg_forecast = agg_df['ensemble'].mean()
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric(
                        label="🎯 Total Forecast",
                        value=f"{total_forecast:,.0f}",
                        help="Total ensemble forecast sessions"
                    )
                with summary_col2:
                    st.metric(
                        label="📊 Average per Period", 
                        value=f"{avg_forecast:,.0f}",
                        help=f"Average {aggregation_type.lower()} forecast"
                    )
                with summary_col3:
                    st.metric(
                        label="📅 Periods",
                        value=f"{len(agg_df)} {aggregation_type.lower()}",
                        help="Number of forecast periods"
                    )
                
                st.divider()
                
                # Always show plot when we have data and should_show_data is True
                should_show_plot = False
                
                if selected_level == "Overall":
                    should_show_plot = True
                elif selected_level in ["Company", "Program", "State"]:
                    # Show plot only for specific selections, not "All"
                    if ((selected_level == "Company" and selected_company != "All") or
                        (selected_level == "Program" and selected_program != "All") or  
                        (selected_level == "State" and selected_state != "All")):
                        should_show_plot = True
                else:
                    # For combination levels, show plot only when specific combinations are selected
                    if not show_aggregated_level_metrics:
                        should_show_plot = True
                
                if should_show_plot and len(agg_df) > 0 and 'ds' in agg_df.columns:
                    fig = go.Figure()
                    
                    # Show prediction interval first (behind the line)
                    if 'ensemble_lo' in agg_df.columns and 'ensemble_hi' in agg_df.columns:
                        fig.add_trace(go.Scatter(
                            x=list(agg_df['ds']) + list(agg_df['ds'][::-1]),
                            y=list(agg_df['ensemble_hi']) + list(agg_df['ensemble_lo'][::-1]),
                            fill='toself',
                            fillcolor='rgba(155, 89, 182, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=True,
                            name='90% Prediction Interval'
                        ))
                    
                    # Show ensemble forecast line
                    fig.add_trace(go.Scatter(
                        x=agg_df['ds'],
                        y=agg_df['ensemble'],
                        mode='lines+markers',
                        name='Ensemble Forecast',
                        line=dict(color='#9B59B6', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title=f"📈 Ensemble Forecast with 90% Prediction Intervals - {title_suffix}",
                        xaxis_title="Date",
                        yaxis_title="Forecasted Sessions",
                        height=500,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom", 
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, width="stretch")
                
                                                # Create enhanced forecast table for selected combination
                st.subheader("📋 Forecast Data Table")
                
                # Prepare display table with better formatting
                display_table = agg_df.copy()
                
                if aggregation_type == "Monthly":
                    display_table['Period'] = display_table['ds'].dt.strftime('%Y-%m')
                    display_table['Period Type'] = 'Month'
                else:
                    display_table['Period'] = display_table['ds'].dt.strftime('%Y-%m-%d')
                    display_table['Period Type'] = 'Week'
                
                display_table['Ensemble Forecast'] = display_table['ensemble'].apply(lambda x: f"{x:,.0f}")
                
                # Add interval columns with forecast in the middle
                table_cols = ['Period', 'Period Type']
                if 'ensemble_lo' in display_table.columns and 'ensemble_hi' in display_table.columns:
                    display_table['Lower Bound'] = display_table['ensemble_lo'].apply(lambda x: f"{x:,.0f}")
                    display_table['Upper Bound'] = display_table['ensemble_hi'].apply(lambda x: f"{x:,.0f}")
                    table_cols.extend(['Lower Bound', 'Ensemble Forecast', 'Upper Bound'])
                else:
                    table_cols.append('Ensemble Forecast')
                
                # Add cumulative at the end
                display_table['Cumulative'] = display_table['ensemble'].cumsum().apply(lambda x: f"{x:,.0f}")
                table_cols.append('Cumulative')
                
                # Select and reorder columns for display
                display_table_final = display_table[table_cols].reset_index(drop=True)
                
                # Display the table with better styling
                st.dataframe(
                    display_table_final, 
                    width="stretch",
                    height=400,
                    hide_index=True
                )
                
            elif show_aggregated_level_metrics and selected_level != "Overall":
                    st.info("💡 Select specific combinations to see forecast plots")
                
            elif should_show_data:
                st.warning("No data available for the selected combination.")
            else:
                st.info("💡 Complete the combination selection to see forecast plots")
        
        except Exception as e:
            st.error(f"Error loading forecast file: {e}")
            st.write(f"File: {latest_forecast_file}")
    
    else:
        st.info("No forecast files found. Please add forecast files to the data folder.")



# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Forecasting Dashboard v1.0 (Enhanced) | Built with Streamlit</div>",
    unsafe_allow_html=True
)