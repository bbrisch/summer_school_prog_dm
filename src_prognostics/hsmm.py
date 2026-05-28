import himap
import numpy as np
from itertools import zip_longest


class CustomHSMM(himap.base.GaussianHSMM):
    def prognostics_pdf(self, X, support_max=100):

        state_seq, _ = self.predict(X)
        state_seq = list(state_seq)

        rul = self.RUL2(state_seq, support_max, 1)
        # rul, _,_,_= self.RUL(similar_seq, max_samples, 1)

        fix_array = np.zeros(support_max)
        fix_array[0] = 1

        rul[rul.sum(1) == 0] = fix_array

        return rul

    def prognostic_bounds(self, X, support_max=100, alpha=0.05):

        # get RUL prognostic pdf
        rul = self.prognostics_pdf(X, support_max)

        x_axis = np.arange(support_max)

        # get expected values and quantiles
        expected_value = (x_axis * rul).sum(1)
        lb = [
            np.quantile(x_axis, alpha / 2, weights=p, method="inverted_cdf")
            for p in rul
        ]
        ub = [
            np.quantile(x_axis, 1 - alpha / 2, weights=p, method="inverted_cdf")
            for p in rul
        ]

        return expected_value, lb, ub

    def RUL2(self, viterbi_states, max_samples, equation=1, parametric=False):
        """
        Estimates the Remaining Useful Life (RUL) for a given state history using convolution of duration probabilities.

        Parameters
        ----------
        viterbi_states : numpy.ndarray
            Sequence of Viterbi states representing the history of hidden states.
        max_samples : int
            Maximum length of RUL to consider.
        equation : int, optional
            Equation type for RUL estimation. Default is 1.

        Returns
        -------
        RUL : numpy.ndarray
            RUL probability distribution for each timestep.
        mean_RUL : numpy.ndarray
            Mean RUL for each timestep.
        UB_RUL : numpy.ndarray
            Upper bound of the RUL distribution.
        LB_RUL : numpy.ndarray
            Lower bound of the RUL distribution.
        """
        RUL = np.zeros((len(viterbi_states), max_samples))
        mean_RUL, LB_RUL, UB_RUL = (np.zeros(len(viterbi_states)) for _ in range(3))
        dur = self.dur
        if parametric:
            dur = self.get_weibull()[2]
        prev_state, stime = 0, 0
        n_states = self.n_states

        for i, state in enumerate(viterbi_states):
            first, second = (np.zeros_like(dur[0, :]) for _ in range(2))
            first[1] = second[1] = 1
            cdf_curr_state = np.cumsum(dur[state, :])
            if state == prev_state:
                stime += 1
            else:
                prev_state = state
                stime = 1

            if stime < len(cdf_curr_state):
                d_value = cdf_curr_state[stime]
            else:
                d_value = cdf_curr_state[-1]

            available_states = np.arange(state, n_states - 1)

            for j in available_states:
                if j != available_states[-1]:
                    first = np.convolve(first, dur[j, :])
                    second = np.convolve(second, dur[j + 1, :])

                else:
                    first = np.convolve(first, dur[j, :])

            if equation == 1:
                first_red = np.zeros_like(first)
                first_red = first[stime:]

                # make sure that after subtracting the soujourn time from the pmf of the first term, that it still sums to 1
                if first_red.sum() != 1:
                    first_red[0] = first_red[0] + (1 - first_red.sum())

            else:
                first_red = first

            first_red = first_red * (1 - d_value)
            second = second * d_value

            result = [sum(n) for n in zip_longest(first_red, second, fillvalue=0)]

            if available_states.size > 0 or not self.last_observed:

                RUL[i, :] = [
                    sum(n) for n in zip_longest(RUL[i, :], result, fillvalue=0)
                ]

            elif not available_states.size > 0 and self.last_observed:
                RUL[i, :], 0
                break

        return np.clip(RUL, 0, np.inf)


def predict_hsmm_bounds(
    trajectory: np.array,
    model: CustomHSMM,
    max_support: int = 612,
    alpha: float = 0.005,
) -> list[np.array, np.array, np.array]:
    """
    Function that generates prognostics using an existing HSMM model.
    - trajectory: Array with CMD of a single trajectory
    - model: trained hsmm model
    - max_support: maximum value of RUL used for generating prognostics.
    - alpha: coinfidence level for generating the intervals
    """

    expected, lb, ub = model.prognostic_bounds(trajectory, max_support, alpha=alpha)

    return expected, np.array(lb), np.array(ub)


def predict_hsmm_pdf_staked(
    trajectory: np.array,
    model: CustomHSMM,
    previous_predictions: list,
    max_support: int = 612,
) -> np.array:
    """
    Function that generates pdf prognostics using an existing HSMM model. The predictions are staked to generate the final prognostics.
    - trajectory: Array with CMD of a single trajectory
    - model: trained hsmm model
    - previous_predictions: list with previous predictions.
    - max_support: maximum value of RUL used for generating prognostics.
    """

    pdf_prognopstic = model.prognostics_pdf(trajectory, max_support)
    previous_predictions.append(pdf_prognopstic)
    return previous_predictions
