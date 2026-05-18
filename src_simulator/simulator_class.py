import numpy as np


class degradation_model:
    def __init__(
        self,
        Q: float,
        R: float,
        omega_std: float,
        eta_std: float,
        dt: float,
        voc_params: dict,
    ):

        self.Q = Q
        self.R = R
        self.omega_std = omega_std
        self.eta_std = eta_std
        self.dt = dt

        self.voc_params = voc_params

    def voc(self, soc):
        a = (self.voc_params["v0"] - self.voc_params["vL"]) * np.exp(
            self.voc_params["gamma"] * (soc - 1)
        )
        b = self.voc_params["alpha"] * self.voc_params["vL"] * (soc - 1)
        c = (
            (1 - self.voc_params["alpha"])
            * self.voc_params["vL"]
            * (
                np.exp(-1 * self.voc_params["beta"])
                - np.exp(-1 * self.voc_params["beta"] * np.sqrt(soc))
            )
        )

        return self.voc_params["vL"] + a + b + c

    def next_state(self, soc, I):
        return np.clip(
            soc + I * self.dt / self.Q + np.random.normal(0, self.omega_std, len(soc)),
            0,
            1,
        )

    def observation_eq(self, soc, I):
        return self.voc(soc) + self.R * I + np.random.normal(0, self.eta_std, len(soc))


class SimulatorSimple:
    def __init__(self, N_simu, v_cut, Soc_0, I, model_config):

        self.model = degradation_model(**model_config)

        self.v_cut = v_cut
        self.I = I
        self.Soc_0 = Soc_0
        self.N_simu = N_simu

    def simulate(self, t_0=0):
        t = t_0
        soc_ = self.Soc_0 * np.ones(self.N_simu)
        v_ = self.model.observation_eq(soc_, self.I)

        alive_tray = self.v_cut <= v_

        soc_memo = [soc_]
        v_memo = [v_]

        keep = True
        soc = soc_

        while keep:
            soc = self.model.next_state(soc, self.I)
            v = self.model.observation_eq(soc, self.I)

            soc_memo.append(soc * alive_tray + np.logical_not(alive_tray) * soc_)
            v_memo.append(v * alive_tray + np.logical_not(alive_tray) * v_)

            v_ = v * alive_tray + np.logical_not(alive_tray) * v_
            soc_ = soc * alive_tray + np.logical_not(alive_tray) * soc_
            alive_tray *= self.v_cut <= v

            t += self.model.dt

            keep = False if np.sum(alive_tray) == 0 else True

        self.soc_memo = np.stack(soc_memo)
        self.v_memo = np.stack(v_memo)
