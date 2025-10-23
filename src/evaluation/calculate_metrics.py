
import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np

from src.metrics.metrics import RegressionMetrics   

def calculate_metrics(result_path, save_path):
    metrics = RegressionMetrics()
    result_data = xr.open_dataset(result_path).load()
    preds = []
    targets = []
    for rep_i in result_data.coords["repetition"]:
        for cv_i in result_data.coords["cv"]:
            pred = result_data["prediction"].sel(repetition=rep_i, cv=cv_i, t=0, ).values
            target = result_data["target"].sel().values
            elems_not_nan = np.invert(np.isnan(pred))
            pred = pred[elems_not_nan]
            target = target[elems_not_nan]
            preds.extend(pred)
            targets.extend(target)
    val_metrics = metrics(targets, preds)
    df = pd.DataFrame(val_metrics, index=[0])
    df.to_excel(save_path)

def calculate_metrics_per_cv(result_path, save_path):
    metrics = RegressionMetrics()
    result_data = xr.open_dataset(result_path).load()
    results_metrics_per_cv = {}
    for rep_i in result_data.coords["repetition"]:
        for cv_i in result_data.coords["cv"].values:
            pred = result_data["prediction"].sel(repetition=rep_i, cv=cv_i, t=0, ).values
            target = result_data["target"].sel().values
            elems_not_nan = np.invert(np.isnan(pred))
            pred = pred[elems_not_nan]
            target = target[elems_not_nan]
            results_metrics_per_cv[f"CV_{cv_i}"] = metrics(target, pred)
    df = pd.DataFrame.from_dict(results_metrics_per_cv, orient="index")
    df.to_excel(save_path)


#def calculate_

#calculate_metrics("/home/miranda/Documents/operational/results/test/results_data.nc", "")
