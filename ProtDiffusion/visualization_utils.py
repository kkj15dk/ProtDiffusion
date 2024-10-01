# %%
import torch
import os
import gc
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt
import time

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
def make_logoplot(array, label:str, png_path:str, positions_per_line:int = 60, width:int = 100, ylim:tuple = (-0.1,1.1), dpi:int = 100, characters:str = "-[]XACDEFGHIKLMNOPQRSTUVWY"):
    start_time = time.time()
    assert array.ndim == 2

    amino_acids = list(characters)

    if os.path.exists(png_path): # If the file already exists, skip making the logoplot
        print(f"File already exists: {png_path}")
        return
    else:
        print(f'Started making logoplot for {label}')

    num_positions = array.shape[1]
    num_lines = (num_positions + positions_per_line - 1) // positions_per_line

    fig, axes = plt.subplots(num_lines, 1, figsize=(width, 5 * num_lines), squeeze=False)

    for line in range(num_lines):
        start = line * positions_per_line
        end = min(start + positions_per_line, num_positions)
        
        df = pd.DataFrame(array.T[start:end], columns=amino_acids, dtype=float)
        
        logo = logomaker.Logo(df, 
                              ax=axes[line, 0],
                              color_scheme=make_color_dict(),
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

    end_time = time.time()
    print(f'Finished making logoplot for {label}')
    print(f'Time taken: {end_time - start_time:.2f} seconds')

    return