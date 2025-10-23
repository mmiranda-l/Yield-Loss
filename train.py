import argparse
import os
import time
from pathlib import Path
import xarray as xr
import shutil

from src.utils.logger import setup_logger
from src.datasets.split import create_splits
from src.datasets.util import create_result_array
from src.datasets.dataset import Dataset_Class  
from src.training.trainer import ML_Trainer
from src.evaluation.calculate_metrics import *
from src.evaluation.visualization import *
from src.utils.utils import *

class Main:
    def __init__(self, settings_path):

        self.settings = load_yaml(settings_path)
        self.data_path = self.settings.get("data_path")
        self.out_dir = self.settings.get("output_dir")
        self.experiment_sett = self.settings["experiment"]
        self.experiment_name = self.experiment_sett.get("experiment_name")
        self.experiment_dir = Path(self.settings.get("output_dir")) / self.experiment_name
        self.path_result_data = self.experiment_dir /  "results_data.nc"
        self.setup_experiment_dir()

    def setup_experiment_dir(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        write_yaml(self.settings, self.experiment_dir / (self.experiment_name + ".yaml"))

    def main_train(self):
        model_args = self.settings["model_settings"]
        start_time = time.time()
        data_path = Path(self.settings.get("data_path"))        

        logger = setup_logger(log_file= self.experiment_dir / "out.log")
        logger.info("Start Experiment")
        split_method = self.experiment_sett.get("split_method", "kfold")
        runs = self.experiment_sett.get("runs", 1)
        kfolds = self.experiment_sett.get("kfolds", 2)
        group_key = self.experiment_sett.get("group_key", None)
        stratification_key = self.experiment_sett.get("stratification_key", None)
        create_result_array(data_path, self.path_result_data, num_cv=kfolds)
        with xr.open_dataset(self.path_result_data) as result_data: 
            for r_seed in range(runs): #RUN   
                splits = create_splits(data_path, group_key, stratification_key, method=split_method, n_splits=kfolds, seed=r_seed)
                for cv_i in range(kfolds): #RUN KFold
                    logger.info(f" Starting run {r_seed} and CV {cv_i} of {runs} runs and {kfolds} CVs")
                    train_indices, val_indices = splits[cv_i]

                    result_data["train_indices"].loc[{"index": train_indices, "cv": cv_i, "repetition": r_seed}] = True
                    result_data["val_indices"].loc[{"index": val_indices, "cv": cv_i, "repetition": r_seed}] = True

                    train_dataset = Dataset_Class(data_path=data_path, indices=train_indices, preload=False)
                    val_dataset = Dataset_Class(data_path=data_path, indices=val_indices, preload=False)
                    
                    trainer = ML_Trainer(
                        out_dir=self.experiment_dir, 
                        path_result_data=result_data, 
                        model_args=model_args, 
                        train_dataset=train_dataset, 
                        val_dataset=val_dataset, 
                        cv=cv_i, 
                        rep_i=r_seed, 
                        logger=logger,
                        seed=r_seed
                        )
                    trainer.training_and_validation()   
            self.save_results(result_data=result_data)

    def save_results(self, result_data):
        if self.path_result_data.is_file():
            os.remove(path=self.path_result_data)
        tmp_path = self.experiment_dir / "tmp.nc"
        result_data.to_netcdf(tmp_path, engine="h5netcdf")
        shutil.move(tmp_path, self.path_result_data)

    def main_eval(self):
        eval_sett = self.settings.get("evaluation")
        eval_dir = self.experiment_dir / "validation"
        os.makedirs(eval_dir, exist_ok=True)
        if eval_sett.get("calculate_metrics", True):
            calculate_metrics(self.path_result_data, save_path=eval_dir / "regression_metrics.xlsx")
            calculate_metrics_per_cv(self.path_result_data, save_path=eval_dir / "regression_metrics_per_cv.xlsx")

        if eval_sett.get("visualization", True):
            vis_dir = self.experiment_dir / "visualization" 
            os.makedirs(vis_dir, exist_ok=True)

            prediction_analysis_simulation_loss_overall(result_ds=self.path_result_data, out_dir=vis_dir, len_target_seq=10)
            
            prediction_analysis_per_field(self.path_result_data, save_dir=vis_dir, len_target_seq=10)
            

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    args = arg_parser.parse_args()
    main = Main(args.settings_file)
    main.main_train()
    main.main_eval()
