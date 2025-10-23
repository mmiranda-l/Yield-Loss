from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
import xarray as xr
import random
import numpy as np
import random

def create_splits(data_path, group_key=None, strat_key=None, method="kfold", n_splits=10, seed=0):
    random.seed(seed)
    print(f"Running {method} cross-validation with {n_splits} folds (seed={seed})")

    with xr.open_dataset(data_path) as ds:
        sample_ids = ds.index.values.tolist()  # must stay a list for .sel(index=...)
        random.shuffle(sample_ids)

        grp_vals = ds[group_key].sel(index=sample_ids).values if group_key else None
        strat_vals = ds[strat_key].sel(index=sample_ids).values if strat_key else None

    if method == "group_kfold":
        splitter = GroupKFold(n_splits=n_splits)
        split_gen = splitter.split(sample_ids, groups=grp_vals)

    elif method == "stratified_group_kfold":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_gen = splitter.split(sample_ids, strat_vals, groups=grp_vals)

    else:  # standard kfold
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_gen = splitter.split(sample_ids)

    cv_splits = [
        (np.take(sample_ids, tr_idx).tolist(), np.take(sample_ids, val_idx).tolist())
        for tr_idx, val_idx in split_gen
    ]
    return cv_splits

if __name__ == "__main__":
    splits = create_splits(
        data_path = "/ds-sds/yieldcon/netCDFs/preprocessed_files/SWISS/merged/merge_s2-weather-ET_yield_loss_dense.nc",
        groups="field_shared_name",
        stratification_key=None, 
        method="group_kfold", 
        n_splits=10

    )
    print(len(splits))
