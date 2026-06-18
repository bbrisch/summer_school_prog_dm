import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

LABELS = {
    "opt_pol": "Optimal",
    "do_nothing": "Do nothing",
    "age": "Age-based",
    "prob_thres": "Prob. thresh.",
    "my_policy": "My policy",
}

COLORS = {
    "opt_pol": "black",
    "do_nothing": "tab:blue",
    "age": "tab:green",
    "prob_thres": "tab:orange",
    "my_policy": "tab:red",
}

POSITIONS = {
    "opt_pol": 0,
    "do_nothing": 1,
    "age": 2,
    "prob_thres": 3,
    "my_policy": 4,
}

X_LABELS = {
    "opt_pol": "Opt",
    "do_nothing": "DN",
    "age": "Age",
    "prob_thres": "Prob",
    "my_policy": "My",

}

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "lines.linewidth": 2.5,
        "font.size": 14,
        "figure.figsize": (12, 5),
    }
)


def plot_policy_comparison(results: dict, alpha: float=0.95) -> None:
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    fig, ax = plt.subplots(nrows=1, ncols=2)

    for key, (c, t, cr, var) in results.items():
        err = z * np.sqrt(var)

        ax[0].errorbar(
            x=POSITIONS[key],
            y=[cr],
            yerr=err,
            fmt="o",
            markersize=8,
            markeredgecolor=COLORS[key],
            markeredgewidth=2,
            color=COLORS[key],
            ecolor=COLORS[key],
            elinewidth=2,
            capsize=6,
            capthick=2,
            label=f"{cr:.3f}",
        )

    xticks = [POSITIONS[k] for k in results.keys()]
    xticklabels = [X_LABELS[k] for k in results.keys()]

    ax[0].set_xticks(xticks, xticklabels)
    ax[0].set_xlabel("Policy", labelpad=10)
    ax[0].set_ylabel(r"$\frac{E[C]}{E[T]}$", rotation=0, labelpad=20)

    yticks=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ax[0].set_yticks(yticks, yticks)
    ax[0].legend()
    ax[0].grid()

    for key, (c, t, cr, var) in results.items():
        n_cp = np.sum(abs(c-10)<1e-3)
        cp_ratio = n_cp/c.size
        if cp_ratio > 1e-6: 
            bar_cp = ax[1].bar(
                x=POSITIONS[key],
                height=cp_ratio,
                bottom=0,
                color=COLORS[key],
                label=rf"preventive",
            )
        if n_cp != c.size:
            bar_cc = ax[1].bar(
                x=POSITIONS[key],
                height=1.0-cp_ratio,
                bottom=cp_ratio,
                color=COLORS[key],
                label=rf"corrective",
            )
            for bar in bar_cc:
                bar.set_hatch("xxx")

    xticks = [POSITIONS[k] for k in results.keys()]
    xticklabels = [X_LABELS[k] for k in results.keys()]

    ax[1].set_xticks(xticks, xticklabels)
    ax[1].set_xlabel("Policy", labelpad=10)
    ax[1].set_ylabel(r"$c_p$ vs. $c_c$", rotation=90, labelpad=0)

    yticks = np.linspace(0, 1, 6)
    ax[1].set_yticks(yticks, yticks)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.01, 0.99))
    ax[1].grid()
    plt.show()
    return