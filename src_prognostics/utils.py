import os
import pickle as pkl
import numpy as np
from numpy import ndarray
import pandas as pd


def load_data(file_name: str) -> pd.DataFrame:
    """
    Function for opening prognostics dataset
    - file_name: type of file, can be either train, validation or test
    """
    assert file_name in [
        "train",
        "validation",
        "test",
    ], 'File name can be either "train", "validation" or "test" '

    return pd.read_parquet(
        os.path.join("src_simulator", "data", f"{file_name}.parquet")
    )


def save_prognostics(prognostics: list, file_name: str):
    path = os.path.join("himap_results", "prognostics")
    os.makedirs(path, exist_ok=True)

    assert file_name in [
        "train",
        "validation",
        "test",
    ], 'File name can be either "train", "validation" or "test" '

    with open(os.path.join(path, file_name + ".pkl"), "wb") as f:
        pkl.dump(prognostics, f)

    print(f"{file_name} saved succesfully!")


def load_prognostics(file_name: str):
    if ".pkl" not in file_name:
        file_name = f"{file_name}.pkl"
    path = os.path.join("himap_results", "prognostics", file_name)

    if not any([mode in file_name for mode in ["train", "validation", "test"]]):
        raise RuntimeError(
            f'File name {file_name} has to be either "train", "validation" or "test"!'
        )

    with open(path, "rb") as f:
        prognostics = pkl.load(f)
    return prognostics


def format_data(df) -> dict:
    """
    Funtion for formatting data in the HIMAP schema.
    Input:
        - df: Pandas dataframe
    """

    output_dict = dict()
    traj_counter = 1

    # Iterate through all the experiments
    exp_ids = df.exp_id.unique()

    max_len = 0

    for exp in exp_ids:
        df_ = df[df.exp_id == exp]
        hi = df_.hi.values.flatten().tolist()
        hi.append(0.34)  # Add last value to enhance prognosability (required by HIMAP)
        hi = np.array(hi) + 1e-3
        hi = hi.flatten()

        # Update the maximum length
        if max_len < len(hi):
            max_len = len(hi)

        # Fill the output dict
        output_dict.update({f"traj_{traj_counter}": hi})
        traj_counter += 1

    return output_dict, max_len

def list_to_nan_padded_array(arr_list: list[ndarray]) -> ndarray:
    """
    Converts a list of 2D-arrays with different lengths into
    a single 3D-array with fixed length, but padded with nan values

    Args:
        arr_list: list of N 2D arrays with sizes (Ti, D)
    
    Returns:
        nan_array (N, Tmax, D)
    """
    # 1. Determine dimensions
    num_comps = len(arr_list)
    max_rows = np.max([a.shape[0] for a in arr_list])
    num_cols = arr_list[0].shape[1] # Assumes all matrices have the same number of columns
    
    # 2. Initialize a 3D array full of NaNs
    nan_array = np.full((num_comps, max_rows, num_cols), np.nan)
    
    # 3. Fill the array slice-by-slice
    for k in range(num_comps):
        nan_array[k, :arr_list[k].shape[0], :] = arr_list[k]
    return nan_array


def calc_M(R: ndarray, R_perfect: ndarray, var_R: ndarray) -> list[ndarray]:
    """
    Calculates the percentage-wise deviation from the perfect policy
    as well as the variance.
    """
    M = (R - R_perfect) / R_perfect
    var_M = var_R / R_perfect**2
    return M, var_M
