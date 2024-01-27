from typing import Dict, List, Tuple
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
    regular_spacing: bool,
    missing_pixel_code: float = np.nan,
    invalid_pixel_code: float = np.inf,
    sfc: Dict = None,
    other_colnames: List[str] = None,
    other_defaults: List = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Sorts the rows of a data table using scurvy
    
    :param df: table with columns `x_colname`, `y_colname`, and `val_colname`
    :param x_colname: name of table column w/ horizontal coords. (eg longitude)
    :param y_colname: name of table column w/ vertical coords. (eg latitude)
    :param val_colname: name of table column w/ property values (eg rainfall)
    :param regular_spacing: are the data points on a grid?
    :param missing_pixel_code: value assigned to missing data pixels
    :param invalid_pixel_code: value assigned to invalid pixels
    :param sfc: output of `surface_filling_curve` applied to `df`
    :param other_colnames: additional names of columns to keep in sorted table
    :param other_defaults: default values for additional columns
    :return: (1) sorted table with `x_colname`, `y_colname`, and `val_colname`;
             (2) surface filling curve object.
             If df has invalid values, then the number of rows of the output
             table may exceed that of the input table; if so, the values of
             other_colnames will equal the defaults provided
             (if other_defaults is not provided, the table is not augmented)
    """
    if sfc is None:
        data, ydim, xdim = convert_df_to_2d_array(
            df=df, 
            x_colname=x_colname,
            y_colname=y_colname, 
            val_colname=val_colname,
            regular_spacing=regular_spacing)
        sfc = surface_filling_curve(
            data=data,
            y=ydim["y"],
            x=xdim["x"],
            missing_pixel_code=missing_pixel_code,
            invalid_pixel_code=invalid_pixel_code,
            verbose=False)

    i = sfc["path"] % len(sfc["y"])
    j = sfc["path"] // len(sfc["y"])
    dd = pd.DataFrame({
        "idx": np.arange(len(i)),
        x_colname: j * ydim["resolution"] + xdim["min"],
        y_colname: i * xdim["resolution"] + ydim["min"],
        val_colname: sfc["data"][i, j]
    })
    
    if other_colnames is not None and len(other_colnames) > 0:
        if other_defaults is not None and \
        len(other_colnames) != len(other_defaults):
            raise ValueError(
                "`other_colnames` must have the same length as `other_defaults`")
        
        nearest = []
        for x, y in zip(dd[x_colname], dd[y_colname]):
            k = np.argmin((x - df[x_colname])**2 + (y - df[y_colname])**2)
            dx = np.abs(x - df[x_colname][k])
            dy = np.abs(y - df[y_colname][k])
            if dx <= xdim["resolution"] and dy <= ydim["resolution"]:
                nearest.append(k)
            else:
                nearest.append(-1)
        if other_defaults is None:
            dd = dd[nearest != -1]
            for colname in other_colnames:
                dd[colname] = [df.loc[k, colname] for k in nearest]
        else:
            for colname, default in zip(other_colnames, other_defaults):
                dd[colname] = [df.loc[k, colname] if k > -1 else default
                               for k in nearest]
    return dd, sfc


def sort_array(
    data: npt.NDArray,
    x: npt.NDArray,
    y: npt.NDArray,
    val_colname: str,
    missing_pixel_code: float = np.nan,
    invalid_pixel_code: float = np.inf,
    sfc: Dict = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Sorts the pixels in a 2D array using surface filling curves
    
    :param data: input 2D array
    :param x: horizontal coords. (eg longitude), len(x)=data.shape[1]
    :param y: vertical coords. (eg latitude), len(y)=data.shape[0]
    :param val_colname: name of property in `data` (eg rainfall)
    :param missing_pixel_code: value assigned to missing data pixels
    :param invalid_pixel_code: value assigned to invalid pixels
    :param sfc: output of `surface_filling_curve` applied to `data`
    :return: (1) table with rows sorted according to surface filling curve;
             (2) surface filling curve dict
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

    return sorted_df, sfc
