import pandas as pd
import numpy as np
import seaborn as sns
import os
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
def set_plot_default():
    mpl.rcdefaults()
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.rcParams['axes.grid'] = False
    mpl.rcParams.update({
        "text.usetex": False,
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

    point_map = {row["Name"]: (row["Position X"], row["Position Y"], row["Position Z"]) for _, row in df.iterrows()}
    distances = {}
    
    for pair in pairs:
        if pair[0] in point_map and pair[1] in point_map:
            point1 = point_map[pair[0]]
            point2 = point_map[pair[1]]
            distances[pair] = euclidean(point1, point2)
    return distances

# Plot the distance matrix
def plot_distances_bar(distances, title="Distances Between Point Pairs"):
    """
    Plots the distances between point pairs as a bar plot.
    """
    if not distances:
        print("No distances to plot.")
        return

    # Prepare data for plotting
    pairs = [f"{pair[0]}-{pair[1]}" for pair in distances.keys()]
    values = list(distances.values())

    # Create the plot
    fig, ax = plt.subplot()
    sns.barplot(x=pairs, y=values, palette="viridis")
    plt.title(title, fontsize=14)
    plt.xlabel("Point Pairs", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    return fig

def plot_distances(distances, title="Distances Between Point Pairs"):
    """
    Plots the distances between point pairs as a scatter and box plot.
    """
    if not distances:
        print("No distances to plot.")
        return

    # Prepare data for plotting
    pairs = [f"{pair[0]}-{pair[1]}" for pair in distances.keys()]
    values = list(distances.values())

    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(x=pairs, y=values, ax=ax[0], size=values, sizes=(40, 100))
    sns.boxplot(y=values, ax=ax[1])
    plt.suptitle(title, fontsize=14)
    ax[0].set_xlabel("Point Pairs", fontsize=12)
    ax[0].set_ylabel("Distance", fontsize=12)
    ax[0].legend(loc="upper left")
    ax[0].grid(True)
    ax[1].set_xlabel("Point Pairs", fontsize=12)
    ax[1].set_ylabel("Distance", fontsize=12)
    #show average distance
    plt.xticks(rotation=45, ha="right")
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