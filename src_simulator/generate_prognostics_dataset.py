# %%  Imports
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulator_class import SimulatorSimple


# %% Load configurations for the simulator
with open("sim_config.yml", "rb") as f:
    sim_config = yaml.safe_load(f)


# %% Generate simulations

# Memory for the simulated trajectories
processed_data = []
load_column = []
id_column = []

# Loading profiles
loading_profiles = sim_config["loading_profiles"]

exp_id = 0
for i, load in enumerate(loading_profiles):

    # Initialize simulator
    sim = SimulatorSimple(
        sim_config["n_simu"],
        sim_config["v_cut"],
        sim_config["soc_0"],
        load,
        sim_config["model_params"],
    )

    # Generate simulations
    sim.simulate()

    # Post process the simulations
    for s in sim.v_memo.T:
        s = s[
            sim_config["v_cut"] * 1.1 < s
        ]  # Eliminate data that falls under the threshold
        s = (s - s[0]) / s[0]  #

        processed_data.append(s)
        id_column.append(np.ones_like(s) * exp_id)
        load_column.append(np.ones_like(s) * i)

        exp_id += 1


# %% Generate a dataframe with the columns
cm_column = np.concatenate(processed_data, 0)  # column of condition monitoring data
load_type_column = np.concatenate(load_column, 0)
exp_id_column = np.concatenate(id_column, 0)

df = pd.DataFrame(
    np.stack([cm_column, load_type_column, exp_id_column], 1),
    columns=["hi", "load", "exp_id"],
)

# Save dataframe as .parquet
df.to_parquet("prognostics_data.parquet")
