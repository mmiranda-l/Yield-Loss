import numpy as np
import xarray as xr
import copy

from src.utils.utils import load_data

def create_result_array(dataset_path, path_result_data, num_cv: int=10, num_rep: int=1, len_pred_sequence: int=55):

    ds = load_data(dataset_path)
    shape = (len(ds.coords["index"].values), num_cv, len_pred_sequence, num_rep)
    dims = ["index", "cv", "t", "repetition"]
    coords = {
        "index": ds.coords["index"].values,
        "cv": np.arange(num_cv, dtype=np.uint16),
        "t": np.arange(len_pred_sequence, dtype=np.uint16),
        "repetition":  np.arange(num_rep, dtype=np.uint16),
    }       
    pred = np.full(shape, fill_value=np.nan, dtype=np.float32)
    tgt_seq = np.full(shape, fill_value=np.nan, dtype=np.float32)
    regularization = np.full(shape, fill_value=np.nan, dtype=np.float32)
    gdseq = np.full(shape, fill_value=np.nan, dtype=np.float32)
    yl = np.full(shape, fill_value=np.nan, dtype=np.float32)

    empty_predictions = xr.DataArray(data=pred, dims=dims, coords=coords)
    ds["prediction"] = empty_predictions

    ky = np.full(shape, fill_value=np.nan, dtype=np.float32)            
    empty_ky = xr.DataArray(data=ky, dims=dims, coords=coords)
    ds["ky"] = empty_ky

    eta = np.full(shape, fill_value=np.nan, dtype=np.float32)            
    empty_eta = xr.DataArray(data=eta, dims=dims, coords=coords)
    ds["eta"] = empty_eta

    # empty_tgt_seq = xr.DataArray(data=tgt_seq, dims=dims, coords=coords)  
    # ds["pred_tgt_sequence"] = empty_tgt_seq 

    # empt_regularization = xr.DataArray(data=regularization, dims=dims, coords=coords)
    # ds["regularization"] = empt_regularization

    # empt_gdseq = xr.DataArray(data=gdseq, dims=dims, coords=coords)
    # ds["gd_seq"] = empt_gdseq

    empy_yl = xr.DataArray(data=yl, dims=dims, coords=coords)
    ds["yl"] = empy_yl

    indices_init = np.full(shape, fill_value=False, dtype=bool)
    indices_init_da = xr.DataArray(data=indices_init, dims=dims, coords=coords)
    ds["train_indices"] = copy.deepcopy(indices_init_da)
    ds["val_indices"] = copy.deepcopy(indices_init_da)
    
    ds.to_netcdf(path_result_data, engine="h5netcdf")
