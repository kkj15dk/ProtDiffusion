# %%
import torch
import os
import gc
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt

from logomaker.src.colors import get_color_dict

# %%
def make_color_dict(color_scheme:str = 'weblogo_protein', cs:str = "-[]XACDEFGHIKLMNOPQRSTUVWY"):
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
def make_logoplot(array, label:str, png_path:str, positions_per_line:int = 64, width:int = 100, ylim:tuple = (-0.2,1.2), dpi:int = 100, characters:str = "-[]XACDEFGHIKLMNOPQRSTUVWY"):
    assert array.ndim == 2

    amino_acids = list(characters)

    if os.path.exists(png_path): # If the file already exists, return the path.
        print(f"File already exists: {png_path}")
        return png_path
    else:
        print(f'Started making logoplot for {label}')

    num_positions = array.shape[1]
    num_lines = (num_positions + positions_per_line - 1) // positions_per_line
    
    fig, axes = plt.subplots(num_lines, 2, figsize=(width*2, 5 * num_lines), squeeze=False)
    for line in range(num_lines):
        start = line * positions_per_line
        end = min(start + positions_per_line, num_positions)
        
        df1 = pd.DataFrame(array.T[start:end], columns=amino_acids, dtype=float)
        
        logo1 = logomaker.Logo(df1, 
                              ax=axes[line, 0],
                              color_scheme=make_color_dict(),
        )
        
        logo1.style_spines(visible=False)
        logo1.style_spines(spines=['left', 'bottom'], visible=True)
        logo1.ax.set_ylabel("Information?")
        logo1.ax.set_xlabel("Position")
        logo1.ax.set_ylim(*ylim)

        # check if any element in the dataframe is negative
        if df1.values.min() > 0:
            df2 = logomaker.transform_matrix(df1, normalize_values=True) # Not sure why this is needed, the values are already normalized
            logo2 = logomaker.Logo(df2,
                                    ax=axes[line, 1],
                                    color_scheme=make_color_dict(),
            )

            logo2.style_spines(visible=False)
            logo2.style_spines(spines=['left', 'bottom'], visible=True)
            logo2.ax.set_ylabel("Probability")
            logo2.ax.set_xlabel("Position")
            logo2.ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.title(f"{label}")

    # Save the figure as a PNG file
    plt.savefig(png_path, dpi = dpi)
    plt.close(logo1.fig)
    plt.close(fig)
    del logo1
    del axes
    del fig
    if df1.values.min() > 0:
        plt.close(logo2.fig)
        del logo2
    gc.collect()  # Force garbage collection
    
    return png_path