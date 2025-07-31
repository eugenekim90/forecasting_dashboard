import numpy as np
import pandas as pd


# Helper functions for calculating evaluation metrics
def mape_skip0(y, yhat):
    mask = y != 0
    if not mask.any():
        return np.nan
    return np.abs((yhat[mask] - y[mask]) / y[mask]).mean() * 100

def wmape_skip0(y, yhat):
    mask = y != 0
    denom = np.abs(y[mask]).sum()
    return np.nan if denom == 0 else np.abs(yhat[mask] - y[mask]).sum() / denom * 100

def mape_zero_ok(y, yhat):
    """Zeros contribute 0 % error (MAPE0)."""
    denom = np.where(y == 0, 1, y)          # dummy 1 avoids /0 but cancels out
    return np.abs((yhat - y) / denom).mean() * 100

def wmape_zero_ok(y, yhat):
    denom = np.abs(y).sum()
    return 0.0 if denom == 0 else np.abs(y - yhat).sum() / denom * 100

def mae_mse_rmse(y, yhat):
    err = y - yhat
    return np.abs(err).mean(), (err**2).mean(), np.sqrt((err**2).mean())


# ── main routine ───────────────────────────────────────────────────────────────
# Function to calculate metrics for each time series
def series_metrics(df, horizon=13, target='y',
                   models=('TimeGPT','LGB','XGB','RF','ensemble')):
    recs = []
    for uid, g in df.groupby('unique_id'):
        last = g.sort_values('ds').tail(horizon)
        y    = last[target].to_numpy()
        for m in models:
            ŷ      = last[m].to_numpy()
            err    = y - ŷ
            mae    = np.abs(err).mean()
            mse    = (err**2).mean()
            rmse   = np.sqrt(mse)

            # series‐level MAPE (skip zeros, but if no nonzero y → 0%)
            nz = y != 0
            if nz.any():
                mape  = np.abs((ŷ[nz] - y[nz]) / y[nz]).mean() * 100
            else:
                mape  = 0.0

            # series‐level WMAPE (skip zeros, but if denom=0 → 0%)
            denom = np.abs(y[nz]).sum()
            if denom > 0:
                wmape = np.abs(ŷ[nz] - y[nz]).sum() / denom * 100
            else:
                wmape = 0.0

            recs.append({
                'unique_id': uid,
                'model':     m,
                'MAE':       mae,
                'RMSE':      rmse,
                'MAPE':      mape,
                'WMAPE':     wmape
            })

    long = pd.DataFrame(recs)

    # reshape back to wide: unique_id + METRIC_MODEL columns
    wide = (
        long
        .set_index(['unique_id','model'])
        .unstack('model')
    )
    wide.columns = [f'{metric}_{model}' for metric, model in wide.columns]
    return wide.reset_index().replace([np.inf, -np.inf], np.nan)

# Function to summarize the evaluation metrics
def metrics_summary(evaluation_metrics):
    summ = (evaluation_metrics
            .describe()
            .loc[['mean', '50%', 'std', 'min']]
            .rename(index={'50%': 'median'}))
    summ.columns = pd.MultiIndex.from_tuples(
        c.split('_', 1) for c in summ.columns
    )
    summ = summ.sort_index(axis=1, level=[0,1])
    return summ.round(2).applymap(lambda x: f'{x:,.2f}').T