from typing import Dict
import numpy.typing as npt
import pandas as pd
import numpy as np

from .convert import convert_df_to_2d_array
from .surface_filling_curve import surface_filling_curve


def sort_df(
    df: pd.DataFrame,
    x_colname: str,
    y_colname: str,
    val_colname: str,
    missing_pixel_code: float = np.nan,
    invalid_pixel_code: float = np.inf,
    sfc: Dict = None
) -> pd.DataFrame:
    """
    Sorts the rows of a data table using scurvy
    
    :param df: table with columns `x_colname`, `y_colname`, and `val_colname`
    :param x_colname: name of table column w/ horizontal coords. (eg longitude)
    :param y_colname: name of table column w/ vertical coords. (eg latitude)
    :param val_colname: name of table column w/ property values (eg rainfall)
    :param missing_pixel_code: value assigned to missing data pixels
    :param invalid_pixel_code: value assigned to invalid pixels
    :param sfc: output of `surface_filling_curve` applied to `df`
    :return: input table, sorted according to surface filling curve
    """
    if sfc is None:
        data, ydim, xdim = convert_df_to_2d_array(
            df=df, 
            x_colname=x_colname,
            y_colname=y_colname, 
            val_colname=val_colname)
        sfc = surface_filling_curve(
            data=data,
            y=ydim["y"],
            x=xdim["x"],
            missing_pixel_code=missing_pixel_code,
            invalid_pixel_code=invalid_pixel_code,
            verbose=False)

    n_path = len(sfc["path"])
    scurvy_idx = np.arange(n_path)[sfc["path"].argsort()]
    i = (np.round((np.array(df[y_colname]) - ydim["min"]) / ydim["resolution"])).astype(int)
    j = (np.round((np.array(df[x_colname]) - xdim["min"]) / xdim["resolution"])).astype(int)
    k = j * ydim["n_pixels_even"] + i
    df['scurvy_idx'] = scurvy_idx[k]
    
    return df.sort_values(by="scurvy_idx")


def sort_array(
    data: npt.NDArray,
    x: npt.NDArray,
    y: npt.NDArray,
    val_colname: str,
    missing_pixel_code: float = np.nan,
    invalid_pixel_code: float = np.inf,
    sfc: Dict = None
) -> pd.DataFrame:
    """
    Sorts the pixels in a 2D array using surface filling curves
    
    :param data: input 2D array
    :param x: horizontal coords. (eg longitude), len(x)=data.shape[1]
    :param y: vertical coords. (eg latitude), len(y)=data.shape[0]
    :param val_colname: name of property in `data` (eg rainfall)
    :param missing_pixel_code: value assigned to missing data pixels
    :param invalid_pixel_code: value assigned to invalid pixels
    :param sfc: output of `surface_filling_curve` applied to `data`
    :return: table with rows sorted according to surface filling curve
    """
    if sfc is None:
        sfc = surface_filling_curve(
            data=data,
            y=y,
            x=x,
            missing_pixel_code=missing_pixel_code,
            invalid_pixel_code=invalid_pixel_code,
            verbose=False)
    
    raw_m, raw_n = sfc["raw_data"]
    m = sfc["data"].shape[0]

    def get_property(k):
        if k % m < raw_m and k // m < raw_n:
            out = sfc["raw_data"][k % m, k // m]
        else:
            out = invalid_pixel_code
        return out
    
    sorted_df = pd.DataFrame([
        [idx, sfc["y"][k % m], sfc["x"][k // m],
         get_property(k)]
        for idx, k in enumerate(sfc["path"])
    ], columns=["scurvy_idx", y_colname, x_colname])

    return sorted_df
