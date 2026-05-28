import os
import pickle as pkl
import numpy as np
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
    path = os.path.join("himap_results", "prognostics", file_name + ".pkl")
    assert file_name in [
        "train",
        "validation",
        "test",
    ], 'File name can be either "train", "validation" or "test" '

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
