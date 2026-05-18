import os
import numpy as np
import pandas as pd


def load_data() -> pd.DataFrame:
    """
    Function for opening prognostics dataset
    """
    return pd.read_parquet(os.path.join("src_simulator", "prognostics_data.parquet"))


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
