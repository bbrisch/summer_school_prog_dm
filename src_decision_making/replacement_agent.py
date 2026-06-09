import numpy as np
from numpy import ndarray

from src_prognostics.utils import load_prognostics, list_to_nan_padded_array

class ReplacementAgent:
    def __init__(self, cf: dict) -> None:
        self.cf = cf
        self._load_prognostics()
        self._get_tfs()
        self._get_prog()
        return
    
    def _load_prognostics(self) -> None:
        self.d_train = load_prognostics(file_name="train")
        self.d_val = load_prognostics(file_name="validation")
        self.d_test = load_prognostics(file_name="test")
        return
    
    def _get_tfs(self) -> None:
        self.trainval_tfs = np.array([k.shape[0] for k in self.d_train+self.d_val]) + 0.5
        self.test_tfs = np.array([k.shape[0] for k in self.d_test]) + 0.5
        return
    
    def _get_prog(self) -> None:
        self.trainval_prog = list_to_nan_padded_array(arr_list=self.d_train+self.d_val)
        self.test_prog = list_to_nan_padded_array(arr_list=self.d_test)
        
        self.trainval_prog = self.trainval_prog[:, self.cf["Delta_T"]-1::self.cf["Delta_T"]]
        self.test_prog = self.test_prog[:, self.cf["Delta_T"]-1::self.cf["Delta_T"]]
        return

    def var_cr(self, c: ndarray, t: ndarray) -> float:
        """
        Computes the variance of the policy performance.

        Args:
            c: array of costs
            t: array of lifecycle

        Returns:
            Var(E[c]/E[t])
        """
        return (
            1
            / np.size(c)
            * (
                np.var(c) / np.mean(t) ** 2
                + np.mean(c) ** 2 * np.var(t) / np.mean(t) ** 4
                - 2 * np.mean(c) * np.cov(c, t)[0, 1] / np.mean(t) ** 3
            )
        )
    
    def opt_pol(self, tfs: ndarray) -> list:
        """
        Optimal replacement policy: components are replaced at last possible
        replacement time before failure.
        """
        costs = np.zeros(tfs.shape)
        tlcs = np.zeros(tfs.shape)
        # check number of failures before first decision time:
        #   -> c_c costs and tf tlcs
        c_c_ind = np.where(tfs <= self.cf["Delta_T"])[0]
        if np.size(c_c_ind) != 0:
            costs[c_c_ind] = self.cf["c_c"]
            tlcs[c_c_ind] = tfs[c_c_ind]

        # otherwise:
        #   -> c_p costs last decision time before tf as tlcs
        valid_ind = np.where(tfs > self.cf["Delta_T"])[0]
        if np.size(valid_ind) != 0:
            tlcs_cp = (tfs[valid_ind] // self.cf["Delta_T"]) * self.cf["Delta_T"]
            ectr_cp = self.cf["c_p"] / tlcs_cp
            ectr_cc = self.cf["c_c"] / tfs[valid_ind]

            let_fail_ind = ectr_cc < ectr_cp

            costs[valid_ind[let_fail_ind]] = self.cf["c_c"]
            costs[valid_ind[~let_fail_ind]] = self.cf["c_p"]

            tlcs[valid_ind[let_fail_ind]] = tfs[valid_ind[let_fail_ind]]
            tlcs[valid_ind[~let_fail_ind]] = tlcs_cp[~let_fail_ind]
        return costs, tlcs, np.mean(costs) / np.mean(tlcs), self.var_cr(c=costs, t=tlcs)

    # ===========================================================================
    # ==================== Evaluation of heuristic policies =====================
    # ===========================================================================

    def step(
        self,
        states: ndarray,
        acts: ndarray,
        costs: ndarray,
        tlcs: ndarray,
        tfs: ndarray,
        time: int,
    ) -> list:
        """
        Performs a single update step.

        Args:
            states (N,): state array
            acts (N,): action array
            costs (N,): cost array
            tlcs (N,): life time array
            tfs (N,): failure time array
            time: current time

        Returns:
            states (N,): new state array
            costs (N,): new cost array
            tlcs (N,): new life time array
        """
        # check for actions, set states to 0, where acts=1
        a_ind = np.where(acts == 1)[0]
        if np.size(a_ind) != 0:
            costs[a_ind] = self.cf["c_p"]
            tlcs[a_ind] = time
            states[a_ind] = 0

        not_a_ind = acts == 0
        fail = tfs < time + self.cf["Delta_T"]
        # check now for failures
        if (np.sum(fail) > 0) and np.sum(not_a_ind) > 0:
            f_ind = np.where(np.logical_and(fail, not_a_ind))[0]
            if np.size(f_ind) > 0:
                costs[f_ind] = self.cf["c_c"]
                tlcs[f_ind] = tfs[f_ind]
                states[f_ind] = 0
        return states, costs, tlcs

    def eval_pol(
        self,
        pol: callable,
        tfs: ndarray,
        prog: ndarray,
        pol_args: list | None=None,
    ) -> list:
        """
        Evalulation of a heuristic for a given cost ratio. The prognostic input is passed
        as a nan-padded array, where N denotes the number of components, T the maximum
        trajectory length and D the support of the probability mass function.

        Args:
            pol: callable heuristic to evaluate -> should return 0/1 action vector
            tfs (N,): array of failure times
            prog (N, T, D): nan-padded RUL-PMF for all time steps
            opt_mode: boolean flag for optimization handling. Decides whether to return
                all values or only R from the output_checker
            par: potential parameter for some heuristics

        Returns either:
            costs: array of all component costs
            tlcs: array of all component lifetimes
            cr: estimate of the policy performance mean(costs) / mean(tlcs)
            var_cr: estimate of the variance of the policy performance
        """
        # actually failures before Delta_t are not implemented in the heuristics check!!
        # state vector, keeps a record of all replaced / failed components
        states = np.ones(tfs.shape, dtype=bool) # (N,)
        costs = np.zeros(tfs.shape) # (N,)
        tlcs = np.zeros(tfs.shape)  # (N,)
        for k in range(prog.shape[1]):
            # get True/False non-nan indices
            if (np.sum(~np.isnan(prog[:, k, :])) > 0) and (np.size(prog[states, k, :]) > 0):

                if np.sum(np.isnan(prog[states, k, :])) > 0:
                    raise Exception("nan value in prog detected!!")

                # calculate actions -> call individual heuristics
                time = (k + 1) * self.cf["Delta_T"]
                acts = np.zeros(states.shape, bool)
                acts[states] = pol(prog=prog[states, k, :], time=time, pol_args=pol_args)

                s2, c2, t2 = self.step(
                    states=states[states],
                    acts=acts[states],
                    costs=costs[states],
                    tlcs=tlcs[states],
                    tfs=tfs[states],
                    time=time,
                )
                costs[states] = c2
                tlcs[states] = t2
                states[states] = s2

        # evaluate those trajectories, where performed DN at last decision time before failure
        if np.sum(states) != 0:
            raise RuntimeError()
        
        try:
            assert not (costs == 0).any()
            assert not (tlcs == 0).any()
        except AssertionError:
            print("Indices with 0 costs: ", np.where(costs == 0)[0])
            print("Indices with 0 life times: ", np.where(tlcs == 0)[0])

        return costs, tlcs, np.mean(costs) / np.mean(tlcs), self.var_cr(c=costs, t=tlcs)
