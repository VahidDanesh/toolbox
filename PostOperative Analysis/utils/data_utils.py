import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import euclidean


# Set the plot theme
def set_plot_theme():
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.rcParams['axes.grid'] = False
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}"
    })

# Read CSV data
def read_csv(file_path):
    """
    Reads a CSV file containing point coordinates.
    Returns a pandas DataFrame.
    """
    return pd.read_csv(file_path)

# Extract point coordinates
def extract_points(df):
    """
    Extracts the (X, Y, Z) coordinates from the DataFrame.
    Returns a numpy array of shape (n_points, 3).
    """
    return df[["Name", "Position X", "Position Y", "Position Z"]]

# Compute pairwise distances
def point_distance(df, pairs=None):
    """
    Computes the Euclidean distances between pairs of points.
    If `pairs` is None, automatically constructs pairs like (1, 1P), (2, 2P), etc.
    If `pairs` is provided, it should be a list of tuples, e.g., [(1, 1P), (2, 2P)].
    Returns a dictionary with pairs as keys and distances as values.
    """
    if pairs is None:
        pairs = [(f"{i}", f"{i}P") for i in range(1, df.shape[0]//2 + 1)]
    distances = {}
    
    for pair in pairs:
        
        if pair[0] in df.Name.values and pair[1] in df.Name.values:
            p1 = df[df.Name == pair[0]].iloc[:, 1:].values
            p2 = df[df.Name == pair[1]].iloc[:, 1:].values
            distances[pair] = euclidean(p1, p2)
    return distances

# Plot the distance matrix
def plot_distance_matrix(distances, title="Pairwise Distances Between Points"):
    """
    Plots the pairwise distance matrix as a heatmap.
    """
    fig, ax = plt.subplots()
    sns.heatmap(
        distances,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Distance"}
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Point Index", fontsize=12)
    plt.ylabel("Point Index", fontsize=12)
    plt.tight_layout()
    plt.show()
    return fig


# https://zhauniarovich.com/post/2022/2022-09-matplotlib-graphs-in-research-papers/
def save_fig(
        fig: plt.figure, 
        fig_name: str, 
        fig_dir: str, 
        fig_fmt: str,
        fig_size: tuple[float, float] = [6.4, 4], 
        save: bool = True, 
        dpi: int = 1200,
        transparent_png = True,
    ):
    """This procedure stores the generated matplotlib figure to the specified 
    directory with the specified name and format.

    Parameters
    ----------
    fig : [type]
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str
        Format of the figure, the format should be supported by matplotlib 
        (additional logic only for pdf and png formats)
    fig_size : Tuple[float, float]
        Size of the figure in inches, by default [6.4, 4] 
    save : bool, optional
        If the figure should be saved, by default True. Set it to False if you 
        do not want to override already produced figures.
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return
    
    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(
        fig_dir,
        '{}.{}'.format(fig_name, fig_fmt.lower())
    )
    if fig_fmt == 'pdf':
        metadata={
            'Creator' : '',
            'Producer': '',
            'CreationDate': None
        }
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(
            pth, 
            bbox_inches='tight',
            dpi=dpi,
        )
    else:
        try:
            fig.savefig(pth, bbox_inches='tight')
        except Exception as e:
            print("Cannot save figure: {}".format(e)) 