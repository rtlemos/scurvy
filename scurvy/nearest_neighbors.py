from typing import Callable, Dict, List, Tuple

import numpy.typing as npt
import numpy as np
import pandas as pd


def get_nearest_neighbors(
        sfc: Dict,
        num_neighbors: int,
        dist_fun_name: str = "manhattan"
) -> Tuple[List[npt.NDArray], List[npt.NDArray], List[npt.NDArray]]:
    """
    Provides the n nearest neighbors for each valid pixel

    :param sfc: output of `surface_filling_curve`
    :param num_neighbors: number of neighbors
    :param dist_fun_name: name of distance function; one of `euclidean`,
                          `manhattan`, `haversine`
    :return: list of distances (len=num_neighbors) to `num_neighbors` nearest
             neighbors; list of neighbor IDs; list of neighbor pixel values
    """

    valid_data_df = sort_valid_data(sfc=sfc)
    ngb = np.array(valid_data_df["id"])
    y = np.array(valid_data_df["y"])
    x = np.array(valid_data_df["x"])
    obs = np.array(valid_data_df["obs"])

    dist_fun = make_dist_fun(dist_fun_name=dist_fun_name)

    distances = []
    neighbors = []
    observations = []
    for i in range(valid_data_df.shape[0]):
        idx = np.arange(max(0, i - num_neighbors), i)
        neighbors.append(ngb[idx])
        distances.append(dist_fun(x[i], y[i], x[idx], y[idx]))
        observations.append(obs[idx])
    return distances, neighbors, observations


def sort_valid_data(
        sfc: Dict
) -> pd.DataFrame:
    """
    Provides the id, coordinates, and value of all valid pixels in an image

    :param sfc: output of `surface_filling_curve`
    :return: table of valid pixels with columns "id", "y", "x", "obs"
    """

    is_valid = make_validator(missing_pixel_code=sfc["missing_pixel_code"],
                              invalid_pixel_code=sfc["invalid_pixel_code"])

    m, n = sfc["data"].shape
    valid_data_df = pd.DataFrame([
        [idx, sfc["y"][k % m], sfc["x"][k // m],
         sfc["raw_data"][k % m, k // m]]
        for idx, k in enumerate(sfc["path"])
        if is_valid(sfc["raw_data"][k % m, k // m])
    ], columns=["id", "y", "x", "obs"])
    return valid_data_df


def make_validator(
        missing_pixel_code: float,
        invalid_pixel_code: float
) -> Callable[[float], bool]:
    """
    Generates the function that determines whether a pixel is valid or not

    :param missing_pixel_code: code for missing pixels (i.e., pixels that could
                               have data but don't)
    :param invalid_pixel_code: code for invalid pixels (i.e., pixels that cannot
                               have data)
    :return: f such that f(pixel) = True if pixel is valid, False otherwise
    """

    if np.isnan(missing_pixel_code):
        return lambda x: \
            False if np.isnan(x) or x == invalid_pixel_code else True
    elif np.isnan(invalid_pixel_code):
        return lambda x: \
            False if np.isnan(x) or x == missing_pixel_code else True
    else:
        return lambda x: \
            False if x in [invalid_pixel_code, missing_pixel_code] else True


def make_dist_fun(dist_fun_name: str):
    """
    Generates distance function between two pixels

    :param dist_fun_name: one of `euclidean`, `manhattan`, 'haversine`
    :return: f such that f(
    """
    if dist_fun_name == "euclidean":
        return lambda x1, y1, x2, y2: \
            np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    elif dist_fun_name == "manhattan":
        return lambda x1, y1, x2, y2: \
            np.abs(x1 - x2) + np.abs(y1 - y2)
    elif dist_fun_name == "haversine":
        return lambda x1, y1, x2, y2: \
            6371 * 2 * np.arcsin(
                np.sqrt(np.sin((y1 - y2) * np.pi / 360) ** 2 +
                        np.cos(y1 * np.pi / 180) * np.cos(y2 * np.pi / 180) *
                        np.sin((x1 - x2) * np.pi / 360) ** 2))
