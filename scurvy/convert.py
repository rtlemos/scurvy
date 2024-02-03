from typing import Dict, Tuple, List
import numpy.typing as npt

import numpy as np
import pandas as pd


def convert_df_to_2d_array(
  df: pd.DataFrame,
  x_colname: str, 
  y_colname: str, 
  val_colname: str,
  regular_spacing: bool,
  empty_value: float = np.inf
) -> Tuple[npt.NDArray, Dict, Dict]:
  
    """
    Converts a dataframe to a 2D array

    :param df: table with columns `x_colname`, `y_colname`, and `val_colname`
    :param x_colname: name of table column w/ horizontal coords. (eg longitude)
    :param y_colname: name of table column w/ vertical coords. (eg latitude)
    :param val_colname: name of table column w/ property values (eg rainfall)
    :param regular_spacing: are the data points on a grid?
    :param fill_value: value used to fill empty pixels
    :return: n_vert * n_horiz array of property values, y-coords, x-coords
    """
    if regular_spacing:
        xdim = get_gridded_dim_info(df[x_colname], "x")
        ydim = get_gridded_dim_info(df[y_colname], "y")
    else:
        xdim = get_nongridded_dim_info([df[x_colname], df[y_colname]], "x")
        ydim = get_nongridded_dim_info([df[y_colname], df[x_colname]], "y")
        
    data = np.full((ydim["n_pixels"], xdim["n_pixels"]), empty_value)
    for k in range(df.shape[0]):
        i = int(np.floor((df[y_colname][k] - ydim["min"]) / ydim["resolution"]))
        j = int(np.floor((df[x_colname][k] - xdim["min"]) / xdim["resolution"]))
        # for originally non-gridded data, the next operation obliterates values
        # that fall on the same grid point (same i, j)
        data[i, j] = df[val_colname][k]
    return data, ydim, xdim


def get_gridded_dim_info(ax: npt.NDArray, ax_colname: str) -> Dict:
    """
    Extracts information about one dimension of the dataset (e.g. longitude)
    This function is used for gridded data

    :param ax: dataframe column with information about one dimension
    :param ax_colname: name of dimension
    :return: dict with summary statistics about dimension
    """
    unique_coords = np.unique(ax)
    resolution = np.median(np.diff(unique_coords))
    mn = unique_coords[0]
    mx = unique_coords[-1]
    n_pixels = int(np.round((mx - mn) / resolution) + 1)
    if n_pixels % 2 == 1:
        mx = unique_coords[-1] + resolution
        n_pixels += 1
    return {ax_colname: np.linspace(mn, mx, n_pixels),
            "min": mn, 
            "max": mx, 
            "resolution": resolution,
            "n_pixels": n_pixels}


def get_nongridded_dim_info(
  ax: List[npt.NDArray], 
  ax_colname: str,
  max_num_cells: int = 1000
) -> Dict:
    cutp = np.percentile(ax[1], np.linspace(0, 100, 10))
    min_resolution = np.ptp(ax[0]) / max_num_cells
    data_resolution = np.median([np.median(np.diff(np.unique(
        ax[0][(ax[1] > cutp[i]) & (ax[1] <= cutp[i + 1])]
    ))) for i in range(9)])
    resolution = max(data_resolution, min_resolution)
    mn = np.min(ax[0])
    n_pixels = int(np.ceil((np.max(ax[0]) - mn) / resolution))
    if n_pixels % 2 == 1:
        n_pixels += 1
    mx = mn + (n_pixels - 1) * resolution
    return {ax_colname: np.linspace(mn, mx, n_pixels),
            "min": mn, 
            "max": mx, 
            "resolution": resolution,
            "n_pixels": n_pixels}
  
