import torch
import os
import gc
import numpy as np
import pandas as pd
import logomaker
import matplotlib.pyplot as plt


@torch.no_grad()
def make_logoplot(array, label:str, png_path:str, positions_per_line:int = 50, width:int = 100, ylim:tuple = (-0.2,1.2), dpi:int = 100, characters:str = "-[]XACDEFGHIKLMNOPQRSTUVWY"):
    assert array.ndim == 2

    amino_acids = list(characters)

    if os.path.exists(png_path): # If the file already exists, return the path.
        print(f"File already exists: {png_path}")
        return png_path
    else:
        print(f'Started making logoplot for {label}')

    num_positions = array.shape[1]
    num_lines = (num_positions + positions_per_line - 1) // positions_per_line
    
    fig, axes = plt.subplots(num_lines, 1, figsize=(width, 5 * num_lines), squeeze=False)
    for line in range(num_lines):
        start = line * positions_per_line
        end = min(start + positions_per_line, num_positions)
        df = pd.DataFrame(array.T[start:end], columns=amino_acids, dtype=float)
        logo = logomaker.Logo(df, ax=axes[line, 0])
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        logo.ax.set_ylabel("Probability")
        logo.ax.set_xlabel("Position")
        logo.ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.title(f"{label}")

    # Save the figure as a PNG file
    plt.savefig(png_path, dpi = dpi)
    plt.close(logo.fig)
    plt.close(fig)
    del logo
    del axes
    del fig
    gc.collect()  # Force garbage collection
    
    return png_path