import numpy as np
import torch
import torch.optim as optim
import os
import shutil

from pathlib import Path
from src.models.lstm import LSTM_Model_Loss_Seq
from src.metrics.metrics import RegressionMetrics   
from src.metrics.loss import PhysicsLoss, WeightedMSE
from src.training.utils import activate_determinism
from src.utils.utils import load_data

class ML_Trainer:
    def __init__(self, out_dir, path_result_data, model_args, train_dataset, val_dataset, cv, rep_i, logger, seed=0):
        self.out_dir = Path(out_dir)
        self.result_data_ds = load_data(path_result_data) 

        self.model_args = model_args
        self.cv = cv
        self.rep_i = rep_i
        self.logger = logger
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.autocast = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_features = train_dataset.num_features
        self.len_sequence = train_dataset.len_sequence
        self.seq_weights = model_args.get("seq_weights", True)
        self.ignored_target_value_loss = model_args.get("ignored_target_value_loss", None)

        self.epochs = model_args.get("epochs", 10)
        self.len_target_seq = model_args.get("len_target_seq", 10)

        self.max_target_value = torch.max(torch.tensor([train_dataset.max_target_value, val_dataset.max_target_value])).to(self.device)
        activate_determinism(seed)
        num_procs = int(len(os.sched_getaffinity(0)))
        g = torch.Generator()
        g.manual_seed(seed)
        
        self.best_val_loss = np.inf
        self.best_r2_value = -np.inf

        self.metrics = RegressionMetrics()
        self.model = self.setup_model()
        self.logger.info(self.model)
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=model_args.get("learning_rate", 0.001),
            weight_decay=model_args.get("weight_decay", 1e-4)
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=model_args.get("lr_scheduler_factor", 0.5)
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            shuffle=True, 
            batch_size=model_args.get("batch_size", 128), 
            pin_memory=True, 
            num_workers=num_procs,
            persistent_workers=True,
            generator=g,
            worker_init_fn=lambda _: np.random.seed(42)
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            shuffle=False, 
            batch_size=model_args.get("batch_size", 128), 
            pin_memory=True,
            num_workers=num_procs,
            persistent_workers=True, #Keeps workers alive between epochs for faster data loading
            generator=g,
            worker_init_fn=lambda _: np.random.seed(42)
        )
        step = 1 / self.len_target_seq
        self.seq_weights = torch.arange(0 + step, 1 + step, step, device=self.device) if self.seq_weights else None

        self.criterion = WeightedMSE(ignored_target_value=self.ignored_target_value_loss, seq_weights=self.seq_weights)
        self.phy_criterion = PhysicsLoss()
        # self.scaler = torch.cuda.amp.GradScaler() # Enable mixed precision
        self.scaler  = torch.amp.GradScaler(self.device)

    def training_and_validation(self):
        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch}")
            self.training(epoch)
            self.validation(epoch)
            
            self.scheduler.step(self.avg_val_loss)
        
    def training(self, epoch):
        self.model.train()
        running_loss = 0.0
        
        for idx, sample_batch in enumerate(self.train_loader):
            ipt_seq = sample_batch["sample"].type(torch.FloatTensor).to(self.device, non_blocking=True)
            physics = sample_batch["physics"].type(torch.FloatTensor).to(self.device, non_blocking=True)
            target = sample_batch["target"].type(torch.FloatTensor).to(self.device, non_blocking=True)
            pred_mask = self.get_pred_mask(
                 sample_batch["time_step"], ipt_seq.shape[0], ipt_seq.shape[1], self.len_target_seq
             )
            #with torch.amp.autocast("cuda"):
            self.optimizer.zero_grad()
            eta, ky = self.model(ipt_seq)
            rs = list(eta.shape)
            rs[1] = self.len_target_seq
            physics =  physics[:,:,1] 
            etx = torch.reshape(physics[pred_mask], rs)
            ky = torch.reshape(ky[pred_mask], rs) 
            eta = torch.reshape(eta[pred_mask], rs)

            yl = self.calculate_yield_loss(eta=eta, ky=ky, etx=etx)
            prediction = self.max_target_value * (1 - yl)
            target = target.unsqueeze(1).repeat(1, self.len_target_seq) #repeat ground truth over the entire time series
            loss_mse = self.criterion(prediction, target) 
            phy_loss = self.phy_criterion(eta, etx)
            print(loss_mse, phy_loss)
            loss = loss_mse + phy_loss
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.detach() * ipt_seq.size(0)
        self.logger.info(f"Epoch: {epoch} Training Loss: {running_loss / len(self.train_loader.dataset):.2f}")

    def validation(self, epoch):
        self.model.eval()
        running_loss = 0.0
        target_list = []
        output_list = []
        index_list = []
        yl_list = []         
        eta_batches = []
        ky_batches = []
        with torch.no_grad():    
            for sample_batch in self.val_loader:
                ipt_seq = sample_batch["sample"].type(torch.FloatTensor).to(self.device, non_blocking=True)
                physics = sample_batch["physics"].type(torch.FloatTensor).to(self.device, non_blocking=True)
                target = sample_batch["target"].type(torch.FloatTensor).to(self.device, non_blocking=True)
                pred_mask = self.get_pred_mask(
                    sample_batch["time_step"], ipt_seq.shape[0], ipt_seq.shape[1], self.len_target_seq
                )
                #with torch.amp.autocast("cuda"):
                eta, ky = self.model(ipt_seq)
                rs = list(eta.shape)
                rs[1] = self.len_target_seq
                physics =  physics[:,:,1] 
                etx = torch.reshape(physics[pred_mask], rs)
                ky = torch.reshape(ky[pred_mask], rs) 
                eta = torch.reshape(eta[pred_mask], rs)

                yl = self.calculate_yield_loss(eta, ky, etx)
                prediction = self.max_target_value * (1 - yl)
                target_seq = target.unsqueeze(1).repeat(1, self.len_target_seq)
                loss_mse = self.criterion(prediction, target_seq) 
                phy_loss = self.phy_criterion(eta, etx)
                loss = loss_mse + phy_loss
                running_loss += loss.detach() * ipt_seq.size(0)
                # store results in lists
                eta_batches.extend(eta.detach().cpu().numpy().tolist())
                ky_batches.extend(ky.detach().cpu().numpy().tolist())
                target_list.extend(target_seq.detach().cpu().numpy().tolist())
                output_list.extend(prediction.detach().cpu().numpy().tolist())
                index_list.extend(sample_batch["index"])
                yl_list.extend(yl.cpu().numpy().tolist())

        val_metrics = self.metrics(target_list, output_list)  
        val_r2 = val_metrics["R2"]          
        self.avg_val_loss = running_loss / len(self.val_loader.dataset)
        if val_r2 > self.best_r2_value:
            self.logger.info(f"Best Model at Epoch {epoch}, Loss: {self.avg_val_loss:.2f}")
            self.logger.info(f"Validation Loss at Epoch {epoch}: {self.avg_val_loss:.2f}, Metrics: {val_metrics}")
            self.best_val_loss = self.avg_val_loss
            self.best_r2_value = val_r2

            target_seq_ids = list(range(self.len_target_seq)) if self.len_target_seq > 1 else 0
            # store results in xarray
            self.result_data_ds["prediction"].loc[{"cv": self.cv, "repetition": self.rep_i, "index": index_list, "t": target_seq_ids}] = output_list
            self.result_data_ds["ky"].loc[
                    {"index": index_list, "repetition": self.rep_i, "cv": self.cv, "t": target_seq_ids}
                     ] = ky_batches
            self.result_data_ds["eta"].loc[
                    {"index": index_list, "repetition": self.rep_i, "cv": self.cv, "t": target_seq_ids}
                        ] = eta_batches
            self.result_data_ds["yl"].loc[
                {"cv": self.cv, "repetition": self.rep_i, "index": index_list, "t": target_seq_ids}
                    ] = yl_list
            

    @torch.jit.script
    def calculate_yield_loss(eta, ky, etx):
        etx = etx + 1e-10 # avoid division by zero 
        evp_red = 1 - (eta / etx) 
        return ky * evp_red
    
    def yield_response_to_water(self, yl): 
        return self.max_target_value * (1 - yl)
    
    def get_pred_mask(self, time_step, batch_size, time_steps, len_target_seq):
        assert time_steps >= len_target_seq, "Not enough time steps in dataset for defined len_target_seq"
        # Allocate mask on GPU
        pred_mask = torch.zeros((batch_size, time_steps), dtype=torch.bool)
        # Adjust time step to avoid out-of-bounds indexing
        time_step = torch.where(time_step == time_steps - 1, time_steps - 2, time_step)
        # Create a tensor of indices for target sequence length
        offsets = torch.arange(len_target_seq,).unsqueeze(0)  # Shape: (1, len_target_seq)
        # Compute mask indices and apply in a vectorized way
        indices = time_step.unsqueeze(1) - offsets  # Shape: (batch_size, len_target_seq)
        indices = indices.clamp(min=0)  # Prevent negative indices
        pred_mask.scatter_(1, indices, True)
        return pred_mask.to(self.device)
    
    def setup_model(self): 
        model_name = self.model_args.get("model_name")
        hidden_size = self.model_args.get("hidden_size", 128)
        num_layers = self.model_args.get("num_layers", 2)  # number of lstm layers
        dropout = self.model_args.get("dropout", 0.3)
        if model_name == "LSTM_Loss":
            model = LSTM_Model_Loss_Seq(in_channels=self.num_features, 
                                        len_max_seq=self.len_sequence, 
                                        hidden_size=hidden_size, 
                                        num_layers=num_layers, 
                                        dropout=dropout)
        else: raise Exception(f"Model {model_name} is not implemented.")

        return model
    
    def save_results(self):
        tmp_path = self.out_dir / "tmp.nc"
        self.result_data_ds.to_netcdf(tmp_path, engine="h5netcdf")
        if self.path_result_data.is_file():
            os.remove(path=self.path_result_data)
        shutil.move(tmp_path, self.path_result_data)
