

import numpy as np
import pandas as pd


def per_model_intervals_no_nan(
    df: pd.DataFrame,
    model_cols=("TimeGPT","LGB","XGB","RF","ensemble","ensemble_ML"),
    alpha: float = 0.10,              # 90% PI
    nonnegative: bool = True,         # clamp lo >= 0 for counts
    zeros_as_missing: bool = False,   # <-- IMPORTANT: treat 0 as real prediction
    fallback_rel: float = 0.30,       # last-resort half-width = max(fallback_abs, |yhat|*fallback_rel)
    fallback_abs: float = 0.0,        # add a tiny absolute floor if you want (e.g., 0.001)
    fillna_yhat_with: float | None = 0.0  # if yhat is NaN, replace with this to avoid NaNs in outputs
) -> pd.DataFrame:
    """
    Bottom level per-model symmetric prediction intervals:
      lo = yhat_m - q_m, hi = yhat_m + q_m
    q_m comes from (1 - alpha) quantile of |y - yhat_m|.
    Fallbacks ensure no NaNs even when residuals are missing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'unique_id', 'ds', 'y' (actual values), and model prediction columns
    model_cols : tuple
        Column names for model predictions
    alpha : float
        Significance level (0.05 for 95% intervals, 0.10 for 90% intervals)
    nonnegative : bool
        Whether to clamp lower bounds to >= 0 (useful for count data like sessions)
    zeros_as_missing : bool
        Whether to treat 0 predictions as missing values
    fallback_rel : float
        Relative fallback width as fraction of prediction
    fallback_abs : float
        Absolute minimum fallback width
    fillna_yhat_with : float or None
        Value to fill NaN predictions with
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with original columns plus {model}_lo and {model}_hi interval columns
    """
    # Validate input
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    required_cols = ['unique_id', 'ds']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    df = df.copy()

    # 1) Prepare predictions for calibration and output
    #    - preds_out: raw yhat (0 is kept as 0)
    #    - preds_cal: used to compute residuals; 0 kept unless zeros_as_missing=True
    preds_out = df.loc[:, model_cols].apply(pd.to_numeric, errors="coerce")
    preds_cal = preds_out.copy()
    if zeros_as_missing:
        preds_cal = preds_cal.mask(preds_cal == 0.0)

    # 2) Build long residual table |y - yhat_m|
    if "y" not in df.columns:
        # If you truly have no y at all, we will skip residual calibration entirely
        print("Warning: No 'y' column found. Using fallback intervals only.")
        long = pd.DataFrame(columns=["unique_id","model","abs_err"])
    else:
        d = pd.concat([df[["unique_id","y"]].copy(), preds_cal], axis=1)
        d = d[d["y"].notna()]
        rows = []
        for m in model_cols:
            if m not in d.columns:
                print(f"Warning: Model column '{m}' not found in DataFrame")
                continue
            e = d[["unique_id","y", m]].rename(columns={m: "yhat"})
            e = e[e["yhat"].notna()]
            if e.empty:
                continue
            e["model"] = m
            e["abs_err"] = (e["y"] - e["yhat"]).abs()
            rows.append(e[["unique_id","model","abs_err"]])
        long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["unique_id","model","abs_err"])

    # 3) Quantiles: per-(unique_id, model) and per-model global
    if not long.empty:
        q_uid = (long
                 .groupby(["unique_id","model"])["abs_err"]
                 .quantile(1 - alpha)
                 .reset_index()
                 .rename(columns={"abs_err":"q"}))
        q_mod = (long
                 .groupby(["model"])["abs_err"]
                 .quantile(1 - alpha)
                 .reset_index()
                 .rename(columns={"abs_err":"q"}))
    else:
        q_uid = pd.DataFrame(columns=["unique_id","model","q"])
        q_mod = pd.DataFrame(columns=["model","q"])

    # 4) Build output frame and attach intervals per model
    out = df[["unique_id","ds"]].copy()
    
    for m in model_cols:
        if m not in preds_out.columns:
            print(f"Warning: Skipping model '{m}' - not found in DataFrame")
            continue
            
        yhat = preds_out[m].copy()

        # Optionally fill missing yhat so we never propagate NaN
        if fillna_yhat_with is not None:
            yhat = yhat.fillna(fillna_yhat_with)

        # q from (unique_id, model)
        q = out[["unique_id"]].assign(model=m).merge(q_uid, on=["unique_id","model"], how="left")["q"]

        # fallback: model-global q
        if q.isna().any():
            if not q_mod.empty and (q_mod["model"] == m).any():
                global_q = q_mod.loc[q_mod["model"] == m, "q"].iloc[0]
                q = q.fillna(global_q)

        # last-resort fallback: default half-width around yhat
        if q.isna().any():
            default_hw = np.maximum(fallback_abs, np.abs(yhat) * fallback_rel)
            q = q.fillna(default_hw)

        lo = yhat - q
        hi = yhat + q
        if nonnegative:
            lo = np.fmax(lo, 0.0)

        out[f"{m}_lo"] = lo
        out[f"{m}_hi"] = hi

    return out


def test_interval_function():
    """
    Test function to demonstrate usage of per_model_intervals_no_nan
    """
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'unique_id': [f'series_{i//10}' for i in range(n_samples)],
        'ds': pd.date_range('2024-01-01', periods=n_samples, freq='W'),
        'y': np.random.poisson(100, n_samples),  # Actual values
        'TimeGPT': np.random.poisson(95, n_samples),
        'LGB': np.random.poisson(98, n_samples),
        'XGB': np.random.poisson(102, n_samples),
        'RF': np.random.poisson(97, n_samples),
        'ensemble': np.random.poisson(99, n_samples),
        'ensemble_ML': np.random.poisson(101, n_samples)
    })
    
    # Generate intervals
    intervals = per_model_intervals_no_nan(
        sample_data,
        model_cols=["TimeGPT","LGB","XGB","RF","ensemble","ensemble_ML"],
        alpha=0.10,  # 90% prediction intervals
        nonnegative=True,
        zeros_as_missing=False,
        fallback_rel=0.30,
        fallback_abs=0.0,
        fillna_yhat_with=0.0
    )
    
    print("Sample of generated prediction intervals:")
    print(intervals.head())
    print(f"\nOutput shape: {intervals.shape}")
    print(f"Columns: {list(intervals.columns)}")
    
    return intervals


if __name__ == "__main__":
    # Run test if script is executed directly
    test_intervals = test_interval_function()


