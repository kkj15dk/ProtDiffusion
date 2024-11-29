# %%
import torch
import os
import gc
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Union

from logomaker.src.colors import get_color_dict

# %%
def make_color_dict(color_scheme:str = 'weblogo_protein', cs:str = "-[]ACDEFGHIKLMNPQRSTVWY"):
    # get list of characters
    cs1 = np.array([c for c in list('ACDEFGHIKLMNPQRSTVWY')])
    cs2 = np.array([c for c in list(cs)])
    # get color dictionary
    rgb_dict = get_color_dict(color_scheme, cs1)

    # add missing characters to color dictionary
    for c in cs2:
        if c not in rgb_dict:
            rgb_dict[c.upper()] = np.array([0.5, 0.5, 0.5])
            rgb_dict[c.lower()] = np.array([0.5, 0.5, 0.5])

    return rgb_dict

# %%
@torch.no_grad()
def make_logoplot(array: np.ndarray, label:str, png_path:str, characters:str = "-[]ACDEFGHIKLMNPQRSTVWY", positions_per_line:int = 64, width:int = 100, ylim:tuple = (-0.1,1.1), dpi:int = 50):
    assert array.ndim == 2

    amino_acids = list(characters)

    if os.path.exists(png_path): # If the file already exists, skip making the logoplot
        print(f"File already exists: {png_path}")
        return

    num_positions = array.shape[1]
    num_lines = (num_positions + positions_per_line - 1) // positions_per_line

    fig, axes = plt.subplots(num_lines, 1, figsize=(width, 5 * num_lines), squeeze=False)

    for line in range(num_lines):
        start = line * positions_per_line
        end = min(start + positions_per_line, num_positions)
        
        df = pd.DataFrame(array.T[start:end], columns=amino_acids, dtype=float)
        
        logo = logomaker.Logo(df, 
                              ax=axes[line, 0],
                              color_scheme=make_color_dict(cs=characters),
        )
        
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        logo.ax.set_ylabel("Probability")
        logo.ax.set_xlabel("Position")
        logo.ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.title(f"{label}")

    # Save the figure as a PNG file
    plt.savefig(png_path, dpi = dpi)
    plt.close(fig)

    gc.collect()  # Force garbage collection

    return

# %%
def latent_ax(ax: plt.Axes, 
              latent: Union[torch.Tensor, np.ndarray], 
              s: int = 20,
              marker: str = 'o', 
              cmap = cm.get_cmap('viridis'),
              xlabel: str = 'Latent Dimension 1', 
              ylabel: str = 'Latent Dimension 2',
              title: Optional[str] = None, 
):
    assert latent.ndim == 2
    dims = latent.shape[0]
    assert dims == 2, "Latent tensor must have 2 dimensions, not {}".format(dims)
    length = latent.shape[1]

    if title is not None:
        ax.set_title(title)
    if np.any(latent):
        ax.scatter(latent[0, :], latent[1, :], s=s, c=np.arange(length), marker=marker, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_ylim(-4, 4)
    ax.set_xlim(-4, 4)

    return

def plot_latent_and_probs(probs: np.ndarray, 
                        latent: np.ndarray, 
                        characters: str = "-[]ACDEFGHIKLMNPQRSTVWY", 
                        positions_pr_line: int = 64, 
                        width: int = 10, 
                        ylim: tuple = (-0.1,1.1), 
                        line_height: int = 1, 
                        symbol_size: int = 30,
                        path: str = None,
                        pad_to_multiple_of: int = 8,
                        title: str = None,
                        noise_pred: Optional[np.ndarray] = None,
):
    # Get the inputs
    latent_dim = latent.shape[0]
    latent_len = latent.shape[1]
    num_positions = latent_len * pad_to_multiple_of
    num_lines = num_positions // positions_pr_line
    n_latent_plots = latent_dim // 2

    # Plotting parameters
    amino_acids = list(characters)
    last_height_ratio = width / (n_latent_plots * line_height)
    noise_plots = 2 if noise_pred is not None else 1
    height_ratios = [1] * num_lines + [last_height_ratio] * noise_plots

    fig_height = line_height * num_lines + (last_height_ratio * noise_plots * line_height)
    fig = plt.figure(figsize=(width, fig_height), dpi=100)
    if title:
        plt.title(title)
    plt.axis('off')

    gs = fig.add_gridspec(num_lines + noise_plots, n_latent_plots, height_ratios=height_ratios)
    axs = [fig.add_subplot(gs[i,:]) for i in range(num_lines)]

    for i, line in enumerate(range(num_lines)):
        start = line * positions_pr_line
        end = min(start + positions_pr_line, num_positions)
        df = pd.DataFrame(probs.T[start:end], columns=amino_acids, dtype=float)

        logo = logomaker.Logo(df, 
                            ax=axs[i],
                            color_scheme=make_color_dict(cs=characters),
                            figsize=(width, line_height),
        )
        
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left'], visible=True) # , 'bottom'], visible=True)
        logo.ax.set_ylabel("Probability")
        logo.ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False, # labels along the bottom edge are off
        )
        # logo.ax.set_xlabel("Position")
        logo.ax.set_ylim(*ylim)

    # Plot latent
    for i in range(n_latent_plots):
        ax = fig.add_subplot(gs[num_lines, i])
        ax.set_title(f"Latent")
        part_latent = latent[i*2:i*2+2]
        latent_ax(ax=ax, 
                    latent=part_latent, 
                    s = symbol_size,
                    xlabel = f"Latent dimension {i*2 + 1}",
                    ylabel = f"Latent dimension {i*2 + 2}",
        )
        if noise_plots == 2:
            noise_ax = fig.add_subplot(gs[num_lines + 1, i])
            noise_ax.set_title(f"Noise")
            part_noise = noise_pred[i*2:i*2+2]
            latent_ax(ax=noise_ax, 
                        latent=part_noise, 
                        s = symbol_size,
                        xlabel = f"Noise dimension {i*2 + 1}",
                        ylabel = f"Noise dimension {i*2 + 2}",
            )

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':
    pass
    # latent = torch.randn(2, 1024)
    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # latent_ax(ax, latent, 'Test')
    # plt.show()
# %%
