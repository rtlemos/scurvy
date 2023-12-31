from typing import List, Dict, Tuple

import numpy.typing as npt
import numpy as np
import pandas as pd
from scipy import interpolate
import plotnine as p9


def preprocess(
        raw_data: npt.NDArray,
        missing_pixel_code: float,
        invalid_pixel_code: float,
        scipy_interpolation_method: str = "nearest",
        verbose: bool = False
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Recodes missing and invalid data, creates masks, and tweaks data for
    compatibility with surface filling curve method

    :param raw_data: 2D matrix of values
    :param missing_pixel_code: code that denotes missing data
    :param invalid_pixel_code: code for invalid pixels (no interpolation)
    :param scipy_interpolation_method: one of "nearest", "linear", "bicubic"
    :param verbose: log to console?
    :return: processed data set and masks for missing and invalid values
    """

    if missing_pixel_code == invalid_pixel_code:
        raise ValueError(
            "missing_pixel_code can't be equal to invalid_pixel_code")

    imputed_data, missing_mask = interpolate_missing_pixels(
        raw_data=raw_data,
        missing_pixel_code=missing_pixel_code,
        scipy_interpolation_method=scipy_interpolation_method
    )
    recoded_data = recode(
        data=imputed_data,
        invalid_pixel_code=invalid_pixel_code
    )
    invalid_mask = np.isnan(recoded_data)
    data = remove_water(data=recoded_data, water=np.nan)
    islands = find_islands(data=data, water=np.nan)
    n = 0
    while len(islands) != 1:
        if verbose:
            print(f"iteration={n}, number of islands={len(islands)}")
        n += 1
        data = connect_islands(data=data, islands=islands, water=np.nan)
        islands = find_islands(data=data, water=np.nan)
    return data, missing_mask, invalid_mask


def interpolate_missing_pixels(
        raw_data: npt.NDArray,
        missing_pixel_code: float,
        scipy_interpolation_method: str = 'nearest',
        fill_value: float = None
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Imputes missing pixels using a scipy method

    :param raw_data: 2D data set with missing pixels
    :param missing_pixel_code: code for missing pixels
    :param scipy_interpolation_method: 'nearest', 'linear' or 'cubic'
    :param fill_value: default interpolation value outside the
        convex hull of valid pixels (for 'linear' and 'cubic')
    :return: data set with missing values filled in, and missing mask
    """

    j, i = np.meshgrid(np.arange(raw_data.shape[1]),
                       np.arange(raw_data.shape[0]))
    data = raw_data.copy()
    if np.isnan(missing_pixel_code):
        missing_mask = np.isnan(raw_data)
    else:
        missing_mask = raw_data == missing_pixel_code
    data[i[missing_mask], j[missing_mask]] = interpolate.griddata(
        points=(j[~missing_mask], i[~missing_mask]),
        values=data[~missing_mask],
        xi=(j[missing_mask], i[missing_mask]),
        method=scipy_interpolation_method,
        fill_value=np.nanmean(raw_data) if fill_value is None else fill_value)
    return data, missing_mask


def recode(
        data: npt.NDArray,
        invalid_pixel_code: float
) -> npt.NDArray:
    """
    Recodes invalid pixels as Nan

    :param data: data set
    :param invalid_pixel_code: original code for missing pixels
    :return: recoded data set where invalid pixels are coded as np.nan
    """
    if ~np.isnan(invalid_pixel_code):
        data = np.where(data == invalid_pixel_code, np.nan, data)
    return data


def make_is_not_water_fn(water: float):
    if np.isnan(water):
        def not_water(data, s):
            return ~np.isnan(data[s[0], s[1]])
    else:
        def not_water(data, s):
            return data[s[0], s[1]] != water
    return not_water


def remove_water(
        data: npt.NDArray,
        water: float
) -> npt.NDArray:
    """
    Ensures that all 2x2 subsets of data are either NaN or not NaN

    :param data: input 2D array
    :param water: code for invalid pixel (e.g. np.nan)
    :return: data set similar to input, with some water pixels replaced with
             the 2x2 means of the subsets that they are in
    """
    nr, nc = data.shape
    if nr % 2 == 1 or nc % 2 == 1:
        raise ValueError("dataset must have even number of rows and cols")
    if np.isnan(water):
        if not np.any(np.isnan(data)):
            return data
        for i in np.arange(0, nr, 2):
            for j in np.arange(0, nc, 2):
                if (np.any(np.isnan(data[i:(i + 2), j:(j + 2)])) and
                        np.any(~np.isnan(data[i:(i + 2), j:(j + 2)]))):
                    x = np.nanmean(data[i:(i + 2), j:(j + 2)])
                    for ii in np.arange(i, i + 2):
                        for jj in np.arange(j, j + 2):
                            if np.isnan(data[ii, jj]):
                                data[ii, jj] = x
    else:
        if not np.any(data == water):
            return data
        for i in np.arange(0, nr, 2):
            for j in np.arange(0, nc, 2):
                if (np.any(data[i:(i + 2), j:(j + 2)] == water) and
                        np.any(data[i:(i + 2), j:(j + 2)] != water)):
                    dt = data[i:(i + 2), j:(j + 2)].ravel()
                    x = np.mean(dt[dt != water])
                    for ii in np.arange(i, i + 2):
                        for jj in np.arange(j, j + 2):
                            if data[ii, jj] == water:
                                data[ii, jj] = x
    return data


def find_islands(
        data: npt.NDArray,
        water: float
) -> List[npt.NDArray]:
    nr, nc = data.shape
    delta = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # down, up, left, right
    not_water = make_is_not_water_fn(water)
    dt = np.copy(data)

    def explore(y, x):
        # adds valid land pixels to list of sought pixels
        search_cells = [[y + d[0], x + d[1]] for d in delta]
        for s in search_cells:
            if 0 <= s[0] < nr and 0 <= s[1] < nc and not_water(dt, s):
                dt[s[0], s[1]] = water
                seek.append(s)

    islands = []
    for la in range(nr):
        for ln in range(nc):
            if not_water(dt, [la, ln]):  # new island discovered
                island = [[la, ln]]
                seek = []
                dt[la, ln] = water
                explore(la, ln)
                while len(seek) > 0:
                    sla, sln = seek.pop(0)
                    island.append([sla, sln])
                    explore(sla, sln)
                islands.append(np.array(island))

    return islands


def connect_islands(
        data: npt.NDArray,
        islands: List[npt.NDArray],
        water: float
) -> npt.NDArray:
    """
    Creates paths (width=2) between islands

    :param data: input 2D array
    :param islands: list of lats/lons of islands
    :param water: code for invalid pixels
    :return: data set with bridges that connect islands
    """

    island_size = [island.shape[0] for island in islands]
    island_order = np.argsort(island_size)
    # connecting islands, from smallest to largest
    made_connection = False
    for idx in island_order:
        island = islands[idx]
        connection = find_connection(data=data, island=island, water=water)
        if connection["length"] < np.inf:
            la, ln = connection["start"]
            sla, sln = connection["end"]
            r = np.mean(data[island[:, 0], island[:, 1]])  # bridge value
            if la == sla:
                data[la:(sla + 2), ln:(sln + 1)] = r
            else:
                data[la:(sla + 1), ln:(sln + 2)] = r
            made_connection = True
    if not made_connection:
        raise ValueError("Could not connect any islands")
    return data


def find_connection(
        data: npt.NDArray,
        island: npt.NDArray,
        water: float
) -> Dict:
    """
    Seeks the shortest path from an island of valid values and a nearby island

    :param data: 2D array of data with islands of value values (1s)
    :param island: 2-column array of lats and lons of pixels of an island
    :param water: code for invalid pixel
    :return: length + coords of up/down/left/right path that connects 2 islands
    """

    nr, nc = data.shape
    not_water = make_is_not_water_fn(water)

    connection = {"length": np.inf, "start": np.nan, "end": np.nan}
    for direction in range(4):
        if direction in [0, 1]:  # up/down
            if direction == 0:
                delta = 1
                extr = np.argmax
            else:
                delta = -1
                extr = np.argmin
            idx = extr(island[:, 0])
            la, ln = island[idx, :]
            found = False
            sla = la
            while 0 < sla < nr - 1 and not found:
                sla += delta
                found = not_water(data, [sla, ln])
            if found:
                length = abs(la - sla) + 1
                if length < connection["length"]:
                    connection["length"] = length
                    connection["start"] = [min(la, sla), ln]
                    connection["end"] = [max(la, sla), ln]
        else:  # left/right
            if direction == 2:
                delta = -1
                extr = np.argmin
            else:
                delta = 1
                extr = np.argmax
            idx = extr(island[:, 1])
            la, ln = island[idx, :]
            found = False
            sln = ln
            while 0 < sln < nc - 1 and not found:
                sln += delta
                found = not_water(data, [la, sln])
            if found:
                length = abs(ln - sln) + 1
                if length < connection["length"]:
                    connection["length"] = length
                    connection["start"] = [la, min(ln, sln)]
                    connection["end"] = [la, max(ln, sln)]
    return connection


def plot_preprocessing(
        raw_data: npt.NDArray,
        data: npt.NDArray,
        y: npt.NDArray = None,
        x: npt.NDArray = None,
        cmap: str = "inferno",
        background_color: str = "white",
        flip_y: bool = False
) -> p9.ggplot:
    """
    Plots raw and processed data side by side, for comparison

    :param raw_data: original 2D array of data
    :param data: output of `preprocess`
    :param y: y-coordinates (len = raw_data.shape[0])
    :param x: x-coordinates (len = raw_data.shape[1])
    :param cmap: matplotlib colormap
    :param background_color: name of background color
    :param flip_y: flip the plot vertically?
    :return:
    """

    nr, nc = raw_data.shape
    if y is None or x is None:
        no_coordinates = True
        y = np.arange(nr)
        x = np.arange(nc)
    else:
        no_coordinates = False
    if flip_y:
        y = np.flip(y)

    def make_df(d, idx):
        la, lo = np.meshgrid(y, x)
        return pd.DataFrame({"y": la.ravel(),
                             "x": lo.ravel(),
                             "z": np.transpose(d).ravel(),
                             "idx": idx
                             })

    df = pd.concat([make_df(raw_data, "raw"), make_df(data, "processed")],
                   axis=0)
    p = p9.ggplot()
    p += p9.geom_tile(data=df, mapping=p9.aes(x="x", y="y", fill="z"))
    p += p9.scale_fill_cmap(cmap)
    p += p9.facet_grid('. ~ idx')
    p += p9.theme(
        panel_background=p9.element_rect(fill=background_color),
        panel_grid_major=p9.element_blank(),
        panel_grid_minor=p9.element_blank()
    )
    if no_coordinates:
        p += p9.theme_void()
    return p
