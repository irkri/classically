from matplotlib import cm as colormaps

def get_color(
    index: int = 0,
    cmap: str = "tab20b",
) -> tuple[float, float, float]:
    """Returns a RGB color as a tuple using some colormap from
    matplotlib. This method is used to easily retrieve already use
    colors again for coloring a pyplot by indexing different colors.

    Args:
        index (int, optional): The index of the color. Defaults to 0.
        cmap (str, optional): The name of the colormap to use.
            Defaults to 'tab20b'.
    """
    return colormaps.get_cmap(cmap).colors[2:][index-5]
