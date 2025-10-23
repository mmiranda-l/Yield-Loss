#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def plot_dist_bin(ax, y_pred_cont, y_true, add_title=""):
    binwidth = 5 if np.max(y_true) > 50  else 0.5
    bins = np.arange(0, np.max(y_true)+ binwidth, binwidth) 
    ax.hist(y_true, label="Ground Truth", alpha=0.6, bins=100 , color="tab:orange", linewidth=1.2)
    ax.hist(y_pred_cont, label="Prediction", alpha=0.35, bins=100, color="tab:blue", linewidth=1.2 )
    ax.set_title(f"Histogram {add_title}")
    ax.legend(loc="upper right")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_xlim(0)

def plot_true_vs_pred(ax, y_pred_cont, y_true, fig, add_title=""):
    y = np.arange(np.min(y_true), np.max(y_true))
    ax.plot(y, y, "-", color="red")
    #ax.scatter(y_true, y_pred_cont, marker="o", edgecolors='black', s=30, rasterized=True)
    hb = plt.hexbin(y_true, y_pred_cont, gridsize=50, cmap='viridis', bins='log',)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Density')
    ax.set_title(f"Prediction vs ground truth {add_title}")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")


def plot_factors(factors, save_path, factors_to_plot=["ETa", "ETx_sim", "yield_loss", "Ky"]):
    fig = plt.figure(figsize=(10,5))
    colors = {"ETa": "tab:blue", "ETx_sim": "tab:orange", "yield_loss": "tab:red", "Ky": "tab:purple"}

    for factor in factors_to_plot:
        if factor in factors.keys():
            factor_values= get_factor_values(factors[factor][factor])
            if factor in ["yield_loss", "Ky"]: 
                factor_values = (factor_values + 1) / 2
            factor_mean = factor_values.mean(axis=0)
            factor_std = factor_values.std(axis=0)
            plt.plot(factor_mean, label=factor, marker="o", c=colors[factor], alpha=0.8)
            plt.fill_between(range(len(factor_mean)), factor_mean + factor_std, factor_mean - factor_std, alpha=0.1, color=colors[factor] )

    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Values")
    plt.savefig(save_path +".pdf", format="pdf")
    plt.close()

def get_factor_values(factor_values):
    factor = []
    for l in range(len(factor_values)):
        factor.extend(factor_values[l])

    return np.stack(factor)
