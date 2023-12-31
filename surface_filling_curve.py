from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import numpy.typing as npt
from scipy.sparse import issparse, csc_matrix
import numba as nb
import plotnine as p9

from preprocess import preprocess


def surface_filling_curve(
        data: npt.NDArray,
        y: npt.NDArray = None,
        x: npt.NDArray = None,
        missing_pixel_code: float = np.nan,
        invalid_pixel_code: float = np.inf,
        verbose: bool = False
) -> Dict:
    """
    Fits a context-dependent surface filling curve to a dataset

    :param data: 2D matrix of data (e.g. elevation), size nr * nc
    :param y: nr array of y-coordinates
    :param x: nc array of x-coordinates
    :param missing_pixel_code: value assigned to missing data pixels
    :param invalid_pixel_code: value assigned to invalid pixels
    :param verbose: log to console?
    :return: context-dependent surface-filling curve
    """

    if data.shape[0] % 2 == 1 or data.shape[1] % 2 == 1:
        raise ValueError(
            "Data matrix must have an even number of rows and columns")

    if data.shape[0] < 6 or data.shape[1] < 6:
        raise ValueError(
            "Data matrix is too small, must be at least 6x6")

    real_coordinates = True
    if y is None:
        y = np.arange(data.shape[0])
        real_coordinates = False
    elif data.shape[0] != len(y):
        raise ValueError(
            "n_row(data) should match length(y)")

    if x is None:
        x = np.arange(data.shape[1])
        real_coordinates = False
    elif data.shape[1] != len(x):
        raise ValueError(
            "n_col(data) should match length(x)")

    proc_data, missing_mask, invalid_mask = preprocess(
        raw_data=data,
        invalid_pixel_code=invalid_pixel_code,
        missing_pixel_code=missing_pixel_code,
        verbose=verbose)
    dual = make_dual(data=proc_data, y=y, x=x, verbose=verbose)
    connectivity_matrix = make_connectivity_matrix(dual=dual, verbose=verbose)
    spath = make_path(connectivity_matrix=connectivity_matrix, verbose=verbose)

    sfc = {
        "raw_data": data,
        "missing_mask": missing_mask,
        "invalid_mask": invalid_mask,
        "data": proc_data,
        "y": y,
        "x": x,
        "connectivity_matrix": connectivity_matrix,
        "path": spath,
        "missing_pixel_code": missing_pixel_code,
        "invalid_pixel_code": invalid_pixel_code,
        "real_coordinates": real_coordinates
    }
    return sfc


def make_coarse_data(
        data: npt.NDArray,
        y: npt.NDArray,
        x: npt.NDArray,
        verbose: bool
) -> Dict:
    """
    Build a coarser dataset

    :param data: 2D matrix of data (e.g. elevation), size nr * nc
    :param y: nr array of y-coordinates
    :param x: nc array of x-coordinates
    :param verbose: log to console?
    :return: coarser dataset
    """
    if verbose:
        print("Building coarser gridded dataset")

    nr, nc = data.shape
    res = abs(y[1] - y[0])
    c_y = np.linspace(np.max(y) - res / 2,
                      np.min(y) + res / 2,
                      int(len(y) / 2))
    c_x = np.linspace(np.min(x) + res / 2,
                      np.max(x) - res / 2,
                      int(len(x) / 2))
    c_data = np.transpose(
        [[safe_nan_mean(data[(i * 2):((i + 1) * 2), (j * 2):((j + 1) * 2)])
          for i in np.arange(int(np.floor(nr / 2)))]
            for j in np.arange(int(np.floor(nc / 2)))])
    coarse = {"data": c_data, "y": c_y, "x": c_x}
    if verbose:
        print("Coarse dataset built.")
    return coarse


def enforce_m(u: Any) -> csc_matrix:
    """
    Forces a user-provided matrix to be of type sparseMatrix

    :param u: User matrix
    :return: sparseMatrix
    """
    if not issparse(u):
        m = np.array(u)
        nr, nc = m.shape
        xx = np.transpose(m).ravel()
        i = np.tile(np.arange(nr), nc)
        j = np.repeat(np.arange(nc), nr)
        crit = xx != 0
        i = i[crit]
        j = j[crit]
        xx = xx[crit]
        u = csc_matrix((xx, (i, j)), shape=(nr, nc))
    return u


def get_column(m: csc_matrix, col_id: int) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Fetches one column of a sparse matrix

    :param m: sparse matrix
    :param col_id: ID of column
    :return: non-empty rows and non-zero values
    """

    st = m.indptr[col_id]
    en = m.indptr[col_id + 1]
    return m.indices[st:en], m.data[st:en]


def get_col_max(
        m: csc_matrix,
        cols: npt.NDArray[int]
) -> npt.NDArray:
    """
    Computes max and which_max for columns of a sparse matrix

    :param m: sparse matrix
    :param cols: column IDs
    :return: 2 * len(cols) matrix with max and which_max
    """
    arr = []
    for col_id in cols:
        col_i, col_x = get_column(m=m, col_id=col_id)
        if len(col_x) > 0:
            idx = np.argmax(col_x)
            arr.append([col_x[idx], col_i[idx]])
        else:
            arr.append([0, col_id])
    return np.array(arr)


def get_neighbors(
        mat: npt.NDArray,
        verbose: bool
) -> Dict:
    """
    Computes valid nearest neighbors of each valid point on a grid

    :param mat: Matrix of grid points
    :param verbose: log to console?
    :return: Dict with IDs of valid points (old and new) and list (ngb) of
             their neighbors
    """
    if verbose:
        print("Building neighborhood structure for dual graph...")
    nr, nc = mat.shape
    n = nr * nc
    valid = ~np.isnan(np.transpose(mat).ravel())

    old_id = np.array([z for z in range(n) if valid[z]])
    new_id = []
    idx = 0
    for i in range(n):
        if valid[i]:
            new_id.append(idx)
            idx += 1
        else:
            new_id.append(np.nan)

    ngb = []
    for k in old_id:
        sub_ngb = []
        c = int(np.floor(k / nr))
        r = k % nr
        if r == 0:
            points = [0, 1]
        elif r == nr - 1:
            points = [-1, 0]
        else:
            points = [-1, 0, 1]
        if c == 0:
            points.append(nr)
        elif c == nc - 1:
            points = [-nr] + points
        else:
            points = [-nr] + points + [nr]
        for p in points:
            if valid[k + p]:
                sub_ngb.append(k + p)
        ngb.append(sub_ngb)
    neighbors = {
        "old_id": old_id,
        "new_id": new_id,
        "ngb": ngb
    }
    if verbose:
        print("Done.")
    return neighbors


def safe_nan_mean(arr: npt.NDArray) -> np.float32:
    """
    Computes the mean of a numpy array that may be 100% NaNs
    :param arr: numpy array
    :return: NaN if all elements are NaN otherwise nan_mean(arr)
    """
    return np.nan if np.all(np.isnan(arr)) else np.nanmean(arr)


def get_similarity_matrix(
        data: npt.NDArray,
        neighbors: Dict,
        verbose: bool
) -> csc_matrix:
    """
    Computes a square similarity matrix based on a raw rectangular matrix
    with gridded information and possibly NAs

    :param data: NxM matrix with gridded data
    :param neighbors: neighborhood structure of dual graph
    :param verbose: log to console?
    :return: (NxM) * (NxM) sparse similarity matrix
    """
    if data.shape[0] % 2 == 1 or data.shape[1] % 2 == 1:
        raise ValueError("data must have even number of rows and columns")
    if verbose:
        print("Building similarity matrix on dual graph... ")

    i = []
    j = []
    for nid, ngb in zip(neighbors["old_id"], neighbors["ngb"]):
        i.extend(np.repeat(nid, len(ngb)))
        j.extend(ngb)
    i = np.array(i).astype(int)
    j = np.array(j).astype(int)

    nr = int(data.shape[0] / 2)
    ci = np.floor(i / nr).astype(int)
    ri = i % nr
    cj = np.floor(j / nr).astype(int)
    rj = j % nr
    distance = []
    for rri, cci, rrj, ccj in zip(ri, ci, rj, cj):
        if cci != ccj:
            c = min(cci, ccj)
            d = safe_nan_mean(np.array([
                np.abs(data[rri * 2 + 0, c * 2 + 1] -
                       data[rri * 2 + 0, c * 2 + 2]),
                np.abs(
                    data[rri * 2 + 1, c * 2 + 1] -
                    data[rri * 2 + 1, c * 2 + 2])
            ])) - safe_nan_mean(np.array([
                np.abs(data[rri * 2 + 0, c * 2 + 1] -
                       data[rri * 2 + 1, c * 2 + 1]),
                np.abs(
                    data[rri * 2 + 0, c * 2 + 2] -
                    data[rri * 2 + 1, c * 2 + 2])
            ]))
        elif rri != rrj:
            r = min(rri, rrj)
            d = safe_nan_mean(np.array([
                np.abs(data[r * 2 + 1, cci * 2] -
                       data[r * 2 + 2, cci * 2]),
                np.abs(data[r * 2 + 1, ccj * 2] -
                       data[r * 2 + 2, ccj * 2]),
            ])) - safe_nan_mean(np.array([
                np.abs(data[r * 2 + 1, cci * 2] -
                       data[r * 2 + 1, ccj * 2]),
                np.abs(data[r * 2 + 2, cci * 2] -
                       data[r * 2 + 2, ccj * 2])
            ]))
        else:
            d = 0
        distance.append(max(0, d))
    sim_nr = int(np.nanmax(neighbors["new_id"]) + 1)
    with np.errstate(divide='ignore'):
        inv_dist = 1 / np.array(distance)
    s = {
        "i": np.array(neighbors["new_id"])[i],
        "j": np.array(neighbors["new_id"])[j],
        "v": inv_dist,
        "nr": sim_nr,
        "nc": sim_nr
    }
    similarity_matrix = csc_matrix((s["v"], (s["i"], s["j"])),
                                   shape=(s["nr"], s["nc"]))
    return similarity_matrix


@nb.guvectorize(["int64[:],int64[:],float64[:],int64[:],int64[:],int64[:]"],
                "(m), (n), (m), (o) -> (o), (o)")
def _make_minimum_spanning_numba(
        similarity_matrix_i,
        similarity_matrix_p,
        similarity_matrix_x,
        dummy_nr,
        tree_out,
        alt_tree_out
):

    nr = len(dummy_nr) + 1
    sm_i = []
    sm_x = []
    for col_id in np.arange(nr):
        st = similarity_matrix_p[col_id]
        en = similarity_matrix_p[col_id + 1]
        sm_i.append(list(similarity_matrix_i[st:en]))
        sm_x.append(list(similarity_matrix_x[st:en]))

    tree = [0]
    tree_arr = np.array(tree)
    alt_tree = [0]
    for col_id in range(nr):
        k = sm_i[col_id].index(col_id)
        sm_i[col_id].pop(k)
        sm_x[col_id].pop(k)

    # finds max(m) and i -> max(m), for each column of similarity matrix
    max_m = []
    max_i = []
    for col_id in np.arange(nr):
        col_i = sm_i[col_id]
        col_m = sm_x[col_id]
        if len(col_m) > 0:
            mx = max(col_m)
            max_m.append(mx)
            max_i.append(col_i[col_m.index(mx)])
        else:
            max_m.append(0)
            max_i.append(col_id)
    max_i = np.array(max_i)
    max_m = np.array(max_m)

    def get_m(i, j):
        # fetches similarity_matrix[i, j] if not 0
        try:
            out = sm_x[j][sm_i[j].index(i)]
        except Exception:
            out = -1
        return out

    for _ in range(nr - 1):
        m = max_m[tree_arr]
        index_j = tree_arr[np.argmax(m)]
        index_i = max_i[index_j]
        for col_id in tree_arr:
            if get_m(index_i, col_id) != 0:
                try:
                    k = sm_i[col_id].index(index_i)
                    sm_i[col_id].pop(k)
                    sm_x[col_id].pop(k)
                except Exception:
                    pass
                try:
                    k = sm_i[index_i].index(col_id)
                    sm_i[index_i].pop(k)
                    sm_x[index_i].pop(k)
                except Exception:
                    pass

        tree.append(index_i)
        tree_arr = np.array(tree)
        alt_tree.append(index_j)

        # update max(m) and i -> max(m), for relevant columns of similarity mat
        for col_id in tree_arr:
            col_i = sm_i[col_id]
            col_m = sm_x[col_id]
            if len(col_m) > 0:
                mx = max(col_m)
                max_m[col_id] = mx
                max_i[col_id] = col_i[col_m.index(mx)]
            else:
                max_m[col_id] = 0
                max_i[col_id] = col_id

    tree = tree_arr[1:nr]
    alt_tree = np.array(alt_tree)[1:nr]
    for k, v in enumerate(tree):
        tree_out[k] = v
    for k, v in enumerate(alt_tree):
        alt_tree_out[k] = v


def make_minimum_spanning_tree(
        similarity_matrix: csc_matrix,
        verbose: bool
) -> csc_matrix:
    nr = similarity_matrix.shape[0]
    if verbose:
        print(f"Building Minimum Spanning Tree for {nr} points")
    tree = np.zeros(nr - 1, dtype=np.int64)
    alt_tree = np.zeros(nr - 1, dtype=np.int64)
    _make_minimum_spanning_numba(
        similarity_matrix.indices,
        similarity_matrix.indptr,
        similarity_matrix.data,
        tree,
        tree,
        alt_tree
    )

    w = {
        "i": np.concatenate([tree, alt_tree], axis=0),
        "j": np.concatenate([alt_tree, tree], axis=0),
        "v": np.ones(2 * (nr - 1), dtype=np.int32),
        "nr": nr,
        "nc": nr
    }
    if verbose:
        print("MST done")
    return csc_matrix((w["v"], (w["i"], w["j"])), shape=(w["nr"], w["nc"]))


def make_dual(
        data: npt.NDArray,
        y: npt.NDArray,
        x: npt.NDArray,
        verbose: bool
) -> Dict:
    """
    Compute the dual graph object

    :param data: 2D matrix of data (e.g. elevation), size nr * nc
    :param y: nr array of y-coordinates
    :param x: nc array of x-coordinates
    :param verbose: log to console?
    :return: dict of dual object
    """
    coarse = make_coarse_data(data=data, y=y, x=x, verbose=verbose)
    neighbors = get_neighbors(mat=coarse["data"], verbose=verbose)
    similarity_matrix = get_similarity_matrix(data=data,
                                              neighbors=neighbors,
                                              verbose=verbose)
    ms_tree = make_minimum_spanning_tree(similarity_matrix=similarity_matrix,
                                         verbose=verbose)
    dual = {
        "data": coarse["data"],
        "y": coarse["y"],
        "x": coarse["x"],
        "neighbors": neighbors,
        "similarity_matrix": similarity_matrix,
        "ms_tree": ms_tree
    }
    return dual


def make_connectivity_matrix(
        dual: Dict,
        verbose: bool
) -> csc_matrix:
    """
    Build a connectivity matrix based on a minimum spanning tree

    :param dual: dual object
    :param verbose: log to console?
    :return: (4*nr*nc) * (4*nr*nc) sparse matrix of connections in dual tree
    """
    if verbose:
        print("Building connection matrix")

    neighbors = dual["neighbors"]
    ms_tree = dual["ms_tree"]
    nr = len(dual["y"])
    nc = len(dual["x"])
    nr2 = 2 * nr
    nc2 = 2 * nc

    boundary = []
    for i in np.arange(nr * nc):
        ri = i % nr
        ci = int(np.floor(i / nr))
        rri = ri * 2
        cci = ci * 2
        idx = cci * nr2 + rri
        if ~np.isnan(neighbors["new_id"][i]):
            if ri == 0 or np.isnan(neighbors["new_id"][i - 1]):
                # no valid neighbor above
                boundary.append([idx, idx + nr2])
            if ri == nr - 1 or np.isnan(neighbors["new_id"][i + 1]):
                # no valid neighbor below
                boundary.append([idx + 1, idx + nr2 + 1])
            if ci == 0 or np.isnan(neighbors["new_id"][i - nr]):
                # no valid neighbor to the left
                boundary.append([idx, idx + 1])
            if ci == nc - 1 or np.isnan(neighbors["new_id"][i + nr]):
                # no valid neighbor to the right
                boundary.append([idx + nr2, idx + nr2 + 1])
    boundary = np.array(boundary)
    boundary = np.concatenate(
        [boundary, np.stack([boundary[:, 1], boundary[:, 0]], axis=1)], axis=0)

    ij = []
    for ngb, i in zip(neighbors["ngb"], neighbors["old_id"]):
        ii = neighbors["new_id"][i]
        ci = int(np.floor(i / nr))
        ri = i % nr
        for j in ngb:
            if j <= i:
                continue
            jj = neighbors["new_id"][j]
            cj = int(np.floor(j / nr))
            rj = j % nr
            if ri == rj:
                # same row, different columns
                if abs(ci - cj) != 1:
                    raise ValueError(
                        f"ci and cj should differ by only 1 unit: {ci}, {cj}")
                r = ri * 2
                c = min(ci, cj) * 2 + 1
                if ms_tree[ii, jj] == 1:
                    ij.append([c * nr2 + r, (c + 1) * nr2 + r])
                    ij.append([c * nr2 + r + 1, (c + 1) * nr2 + r + 1])
                else:
                    ij.append([c * nr2 + r, c * nr2 + r + 1])
                    ij.append([(c + 1) * nr2 + r, (c + 1) * nr2 + r + 1])
            else:
                # same column, different rows
                if abs(ri - rj) != 1:
                    raise ValueError(
                        f"ri and rj should differ by only 1 unit: {ri}, {rj}")
                r = min(ri, rj) * 2 + 1
                c = ci * 2
                if ms_tree[ii, jj] == 1:
                    ij.append([c * nr2 + r, c * nr2 + r + 1])
                    ij.append([(c + 1) * nr2 + r, (c + 1) * nr2 + r + 1])
                else:
                    ij.append([c * nr2 + r, (c + 1) * nr2 + r])
                    ij.append([c * nr2 + r + 1, (c + 1) * nr2 + r + 1])
    ij = np.array(ij)
    ij = np.concatenate(
        [ij, np.stack([ij[:, 1], ij[:, 0]], axis=1)], axis=0)

    links = np.concatenate([ij, boundary], axis=0)
    c = {
        "i": links[:, 0],
        "j": links[:, 1],
        "v": np.ones(links.shape[0]),
        "nr": nr2 * nc2,
        "nc": nr2 * nc2
    }
    return csc_matrix((c["v"], (c["i"], c["j"])), shape=(c["nr"], c["nc"]))


def get_j(m: csc_matrix):
    j = []
    for c in range(m.shape[1]):
        j.extend([c] * (m.indptr[c + 1] - m.indptr[c]))
    return np.array(j)


def make_path(
        connectivity_matrix: csc_matrix,
        verbose: bool
) -> npt.NDArray[int]:
    """
    Computes a path that connects all the points in a matrix

    :param connectivity_matrix: (4*nr*nc)*(4*nr*nc) conn. matrix in dual tree
    :param verbose: log to console?
    :return: Path that connects all elements in connection matrix
    """
    if verbose:
        print("Building path for dual tree")

    nr = connectivity_matrix.shape[0]
    i = connectivity_matrix.indices
    j = get_j(connectivity_matrix)

    cmax = get_col_max(connectivity_matrix, np.arange(nr))
    st = 0
    for k in np.arange(nr):
        if cmax[k, 0] > 0:
            break
        st += 1
    spath = [st]
    free = np.full(nr, True, dtype=bool)
    free[st] = False
    neighbor = j[i == st]
    crit = free[neighbor]

    while np.sum(crit) > 0:
        point = neighbor[crit][0]
        free[point] = False
        spath.append(point)
        neighbor = j[i == point]
        crit = free[neighbor]

    if verbose:
        print("Finished building path")
    return np.flip(spath)


def plot_path(
        sfc: Dict,
        plot_data: bool = False,
        lat_bounds: Tuple[float, float] = None,
        lon_bounds: Tuple[float, float] = None,
        path_cmap: str = "gnuplot2",
        aspect_ratio: float = 1,
        background: str = "white",
        show_line_scale: bool = True,
        show_fill_scale: bool = True,
        line_width: float = 0.5
) -> p9.ggplot:
    """
    Plots a path

    :param sfc: surface filling curve
    :param plot_data: plot the data?
    :param lat_bounds: y-coordinate bounds (for zooming in)
    :param lon_bounds: x-coordinate bounds (for zooming in)
    :param path_cmap: color map for path
    :param aspect_ratio: aspect ratio of axes
    :param background: background color
    :param show_line_scale: display color bar for line?
    :param show_fill_scale: display color bar for fill?
    :param line_width: path line width
    :return: plot
    """
    spath = sfc["path"]
    y = sfc["y"]
    x = sfc["x"]
    data = np.transpose(sfc["data"]).ravel()

    n = len(spath)
    path_col = np.floor(spath / len(y)).astype(int)
    path_row = (spath % len(y)).astype(int)
    ri = path_row[np.arange(n)]
    ci = path_col[np.arange(n)]
    idx = (np.arange(n) * 10 / (n - 1)).astype(int)

    p = p9.ggplot()
    p += p9.coord_fixed(ratio=aspect_ratio, xlim=lon_bounds, ylim=lat_bounds)
    if plot_data:
        la, lo = np.meshgrid(y, x)
        rst = np.stack([lo.ravel(), la.ravel(), data], axis=1)
        da = pd.DataFrame({"x": rst[:, 0], "y": rst[:, 1],
                           "z": rst[:, 2]}).dropna()
        p += p9.geom_raster(data=da,
                            mapping=p9.aes(x="x", y="y", fill="z"),
                            show_legend=show_fill_scale)
        p += p9.scale_fill_gradient(low="#222222", high="#a9a9a9")

    df = pd.DataFrame({"y": y[ri], "x": x[ci], "path": idx})
    if path_cmap is None:
        p += p9.geom_path(data=df,
                          mapping=p9.aes(x="x", y="y"),
                          show_legend=False,
                          size=line_width)
    else:
        p += p9.geom_path(data=df,
                          mapping=p9.aes(x="x", y="y", color="path"),
                          show_legend=show_line_scale,
                          size=line_width)
        p += p9.scale_color_cmap(path_cmap)
    if not sfc["real_coordinates"]:
        p += p9.theme_void()
    p += p9.theme(
        panel_background=p9.element_rect(fill=background),
        panel_grid_major=p9.element_blank(),
        panel_grid_minor=p9.element_blank())
    return p
