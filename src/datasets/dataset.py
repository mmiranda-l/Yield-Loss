from torch.utils.data import Dataset
import xarray as xr 
import numpy as np
import os
import torch

from src.utils.utils import load_data


class Dataset_Class(Dataset):
    def __init__(self, data_path: str, indices: list = None, fill_value=-1, preload=False):
        super().__init__()
        assert os.path.exists(data_path), f"Dataset file {data_path} not found."
        
        self.data_path = data_path
        self.indices = indices
        self.preload = preload
        self.fill_value = fill_value
        self.ds = load_data(data_path).load()
        self.ds["sample"] = self.ds["sample"].fillna(fill_value)
        self.ds["sample_sim"] = self.ds["sample_sim"].fillna(fill_value)
        self.num_features = self.ds.band.values.size
        self.len_sequence = self.ds.time_step.values.size
        self.max_target_value = self.ds["target"].values.max()

        if self.indices is not None:
            self.ds = self.ds.sel(index=self.indices)

        self.index_ids = list(self.ds.coords["index"].values)
        # Precompute harvesting steps for each index
        self.field_time_steps = {
            idx_id: self.get_time_steps(idx_id) for idx_id in self.index_ids
        }
        if preload:
            print(f"Preloading data for faster training. Only recommended for small datasets.")
            self._preload_data()

    def __len__(self):
        return len(self.ds["sample"])

    def get_time_steps(self, idx):
        time_vals = self.ds["times"].sel(index=idx).values
        valid_times = time_vals[~np.isnat(time_vals)]
        # Ensure chronological order
        if valid_times[0] > valid_times[-1]:
            time_vals = time_vals[::-1]
            valid_times = valid_times[::-1]

        harvest_id = self.ds["harvesting_date"].sel(index=idx).item()
        harvest_date = np.datetime64(self.ds["harvesting_date"].attrs[str(harvest_id)])

        date_diffs = harvest_date - time_vals
        valid_mask = date_diffs >= np.timedelta64(0, "D")

        # Assign large timedelta for invalid entries, then argmin
        adjusted_diffs = np.where(valid_mask, date_diffs, np.timedelta64(9999, "D"))
        return int(np.argmin(adjusted_diffs))

    def __getitem__(self, index):
        if self.preload:
            return self.__getimtem_preload(index)
        else: return self.__getitem_core(index)
    
    def __getitem_core(self, index):
        index_id = self.index_ids[index]
        sample = self.ds["sample"].isel(index=index).values.astype(np.float32)
        target = self.ds["target"].isel(index=index).values.astype(np.float32)
        phys = self.ds["sample_sim"].isel(index=index)
        phys = phys.sel(band=["ETref", "ETc", "ETcadj"]).values.astype(np.float32)
        phys = np.nan_to_num(phys, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value)
        time_step = self.field_time_steps[index_id]
        sample = np.nan_to_num(sample, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value)

        return {
            "index": str(index_id),
            "sample": torch.tensor(sample, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "physics": phys,
            "time_step": time_step
        }
    
    def __getimtem_preload(self, index):

        return {
            "index": self.index_id_list[index],
            "sample": self.sample_list[index],
            "target": self.target_list[index],
            "physics": self.physics_list[index],

            "time_step": self.time_step_list[index]
        }   
    
    def _preload_data(self):
        self.sample_list = []
        self.target_list = []
        self.index_id_list = []
        self.physics_list = []

        self.time_step_list = []
        for i in range(len(self)):
            sample_i = self.__getitem_core(i)
            self.sample_list.append(sample_i["sample"])
            self.target_list.append(sample_i["target"])
            self.index_id_list.append(sample_i["index"])
            self.time_step_list.append(sample_i["time_step"])
            self.physics_list.append(sample_i["physics"])


