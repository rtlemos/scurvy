from typing import Dict
import numpy.typing as npt

import numpy as np
import pandas as pd


def convert_df_to_2d_array(
  df: pd.DataFrame,
  x_colname: str, 
  y_colname: str, 
  val_colname: str
) -> npt.NDArray:
  
    """
    Converts a dataframe to a 2D array

    :param df: table with columns `x_colname`, `y_colname`, and `val_colname`
    :param x_colname: name of table column w/ horizontal coords. (eg longitude)
    :param y_colname: name of table column w/ vertical coords. (eg latitude)
    :param val_colname: name of table column w/ property values (eg rainfall)
    :return: n_vertical * n_horizontal array of property values 
    """
    xdim = get_dim_info(df[x_colname])
    ydim = get_dim_info(df[y_colname])
    data = np.full((ydim["even_n_pixels"], xdim["even_n_pixels"]), np.nan)
    for k in range(df.shape[0]):
        i = int(np.round((df[y_colname][k] - ydim["min"]) / ydim["resolution"]))
        j = int(np.round((df[x_colname][k] - xdim["min"]) / ydim["resolution"]))
        data[i, j] = df[val_colname][k]
    return data


def get_dim_info(coords1d: npt.NDArray) -> Dict:
    """
    Extracts information about one dimension of the dataset (e.g. longitude)

    :param coords1d: dataframe column with information about one dimension
    :return: dict with summary statistics about dimension
    """
    unique_coords = np.unique(coords1d)
    nc = len(unique_coords)
    resolution = np.median(unique_coords[1:nc] - unique_coords[0:(nc - 1)])
    n_pixels = int(np.round((unique_coords[-1] - unique_coords[0]) / resolution + 1))
    even_n_pixels = 2 * ((n_pixels + 1) // 2)
    return {"min": unique_coords[0], 
            "max": unique_coords[-1], 
            "resolution": resolution,
            "n_pixels": n_pixels, 
            "even_n_pixels": even_n_pixels}
  
