import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.utils.utils import *

def plot_et(ax, ETc, ETa):

    ax.plot(ETa, linestyle="--", alpha=1, c="#2461a3", label="$ETa^{pred}$", marker="o")

    ax.plot(ETc, label="$ETx^{sim}$", marker="o", linestyle="--", alpha=0.9, c="tab:orange")
    ax.set_ylabel("ET [mm]")
    ax.legend()

def plot_ky(ax, ky):
    ax.plot(ky, linestyle="--", alpha=0.9, c="#2461a3", label="Ky", marker="o")
    ax.legend(fontsize=15)

def plot(path_out, ETc, ETa, ky):
    fig, ax = plt.subplots(1, 2, figsize=(22, 5))
    plot_et(ax[0], ETc, ETa)
    plot_ky(ax[1], ky)
    plt.tight_layout()
    plt.savefig(path_out, dpi=120)
    plt.close()

def prediction_analysis_simulation_loss_overall(result_ds, out_dir, len_target_seq=10):

    result_ds = load_data(result_ds)
    fig, ax = plt.subplots(1, 2, figsize=(22, 5))
    ETc = result_ds["sample"].sel(band=["ETcadj"]).mean(dim=["index", "band"]).values[-len_target_seq:]
    ky = result_ds["ky"].mean(dim=["index", "cv"]).values[:, -len_target_seq:]
    ETa = result_ds["eta"].mean(dim=["index", "cv"])[:,-len_target_seq:]

    yield_loss = ky * (1 - ETa / ETc)
    plot_et(ax[0], ETc, ETa)

    ax[1].plot(yield_loss.mean(axis=0), linestyle="--", alpha=0.9, c="#93032E", label="yield loss (%)", marker="o")
    ax[1].legend()
    # plot_ky_ndvi(ax[1], ky, ndvi, xticks_labels=xticks_labels, xticks_positions=xticks_positions)
    plt.savefig(out_dir / "prediction_overall.pdf", dpi=120)
    plt.close() 

def plot_predictions_per_field(prediction, target, prediction_eta, save_path):
    x_label = "Time steps before harvesting"
    fig, axs = plt.subplots(1, 6, figsize=(45, 8))
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color="white", alpha=1)
    cmap_m = plt.colormaps["magma"].copy()
    cmap_m.set_bad(color="white", alpha=1)
    vmin = min(np.nanmin(prediction), np.nanmin(target))
    vmax = max(np.nanmax(prediction), np.nanmax(target))
    plot = axs[0].imshow(target, cmap=cmap)
    axs[0].title.set_text(f"Ground Truth {np.nanmean(target):.2f} (t/ha)")
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation='vertical')

    plot = axs[1].imshow(prediction, cmap=cmap)
    axs[1].title.set_text(f"Prediction yield {np.nanmean(prediction):.2f} (t/ha)")
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation='vertical')

    plot = axs[2].imshow(prediction_eta, cmap=cmap_m)
    axs[2].title.set_text(f"Prediction ETa {np.nanmean(prediction_eta):.2f}")
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation='vertical')

    norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plot = axs[3].imshow((prediction - target), cmap=cmap, norm=norm)
    axs[3].title.set_text(f"Relative Error (%)")
    divider = make_axes_locatable(axs[3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(plot, cax=cax, orientation='vertical')

    axs[4].scatter(target.flatten(), prediction.flatten(), alpha=.4, c="#33746B")
    axs[4].plot([vmin, vmax], [vmin, vmax], color="k", ls="-", alpha=.5)
    axs[4].title.set_text(f"Prediction over target")
    axs[4].set_xlabel("Target")
    axs[4].set_ylabel("Prediction")

    prediction = prediction.flatten()
    target = target.flatten()
    pos_nan = np.isnan(prediction)
    prediction = prediction[~pos_nan]
    pos_nan = np.isnan(target)
    target = target[~pos_nan]
    bins = np.histogram(np.hstack((prediction, target)), bins=75)[1]
    axs[5].hist(prediction, bins=bins, fc=(0.14, 0.38, 0.64, 0.7), label="prediction")
    axs[5].hist(target, bins=bins, fc=(0.21, 0.59, 0.44, 0.7), label="target")
    axs[5].legend()
    axs[5].title.set_text("histogram prediction and target")
    axs[5].set_xlabel("yield (%)")
    axs[5].set_ylabel("count (#pixels)")
    fig.tight_layout()

    plt.savefig(save_path)

def prediction_analysis_per_field(result_ds, save_dir, len_target_seq=10): 
    save_dir = Path(save_dir)
    result_ds = load_data(result_ds)
    fsns = result_ds["field_shared_name"].attrs
    phys_dir  = save_dir / "temporal_prediction_phys_per_field"
    yield_dir = save_dir / "yield_prediction_per_field"
    os.makedirs(phys_dir, exist_ok=True)
    os.makedirs(yield_dir, exist_ok=True)

    for fsn in result_ds["field_shared_name"].attrs:
        fsn_name = fsns[fsn]
        index_bool = result_ds["field_shared_name"] == int(fsn)
        indices_fsn = result_ds["index"][index_bool].values
        rows = result_ds["row"].sel(index=indices_fsn).values.tolist()
        cols = result_ds["col"].sel(index=indices_fsn).values.tolist()
        target = np.full((max(rows) + 1, max(cols) + 1), fill_value=np.nan)
        row_i = result_ds["row"].sel(index=indices_fsn).values
        col_i = result_ds["col"].sel(index=indices_fsn).values
        target[row_i, col_i] = result_ds["target"].sel(index=indices_fsn).values
        prediction_eta = np.full((max(rows) + 1, max(cols) + 1), fill_value=np.nan)
        prediction = np.full((max(rows) + 1, max(cols) + 1), fill_value=np.nan)

        prediction_eta[row_i, col_i] = (  
                np.nanmean(result_ds["eta"].sel(index=indices_fsn, repetition=0, t=0).values, axis=-1)
            )
        prediction[row_i, col_i] = (
                np.nanmean(result_ds["prediction"].sel(index=indices_fsn, repetition=0, t=0).values, axis=-1 )
                    # result_ds["prediction"].sel(index=indices_fsn, repetition=0, cv=cv_i, t=0).values
                )
        path_out_yield = str(yield_dir / (fsn_name + ".pdf"))
        plot_predictions_per_field(prediction=prediction, target=target, prediction_eta=prediction_eta, save_path=path_out_yield)

        for cv_i in result_ds.coords["cv"].values:
            if not np.isnan(
                result_ds["prediction"].sel(index=indices_fsn, repetition=0, cv=cv_i, t=0).values
            ).all():
                prediction = np.full((max(rows) + 1, max(cols) + 1), fill_value=np.nan)
                prediction[row_i, col_i] = (
                    np.nanmean(result_ds["prediction"].sel(index=indices_fsn, repetition=0, t=0).values, axis=-1 )
                    # result_ds["prediction"].sel(index=indices_fsn, repetition=0, cv=cv_i, t=0).values
                )
                ETc = result_ds["sample"].sel(band=["ETcadj"]).sel(index=indices_fsn).mean(dim=["index", "band"]).values[-len_target_seq:]
                ky = result_ds["ky"].sel(index=indices_fsn, cv=cv_i).mean(dim=["index"])[:, -len_target_seq:]
                ETa = result_ds["eta"].sel(index=indices_fsn, cv=cv_i).mean(dim=["index"])[:,-len_target_seq:]
                # print(ETa.shape, ETc.shape, ETcadj.shape, ky.shape)

                path_out_phys = str(phys_dir / (fsn_name + ".pdf"))
                plot(path_out=path_out_phys, ETc=ETc, ETa=ETa, ky=ky) 

if __name__ == "__main__":
    result_data = Path("/home/miranda/Documents/operational/repositories/Yield-Loss/out/example_run/example_run/results_data.nc")
    dir_out = Path("/home/miranda/Documents/operational/repositories/Yield-Loss/out/example_run/example_run")
    prediction_analysis_per_field(result_data, dir_out) 
