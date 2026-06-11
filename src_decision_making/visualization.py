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

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.labelsize": 22,
        "legend.fontsize": 15,
        "lines.linewidth": 2.5,
        "font.size": 18,
        "figure.figsize": (12, 7.5),
    }
)


def plot_policy_comparison(results: dict, alpha: float=0.95) -> None:
    z = stats.norm.ppf(1 - (1 - alpha) / 2)

    fig, ax = plt.subplots()

    for key, (c, t, cr, var) in results.items():
        err = z * np.sqrt(var)

        ax.errorbar(
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
    xticklabels = [LABELS[k] for k in results.keys()]

    ax.set_xticks(xticks, xticklabels)
    ax.set_xlabel("Policy", labelpad=15)
    ax.set_ylabel(r"$\frac{E[C]}{E[T]}$", rotation=0, labelpad=25)

    #ax.set_yscale("symlog", linthresh=0.08)
    #yticks = np.array([0.06, 0.08, 0.1, 0.2, 0.4, 0.6])
    yticks=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ax.set_yticks(yticks, yticks)

    ax.legend()
    ax.grid()
    plt.show()
    return

def plot_cp_vs_cc(results: dict) -> None:
    fig, ax = plt.subplots()

    for key, (c, t, cr, var) in results.items():
        n_cp = np.sum(abs(c-10)<1e-3)
        cp_ratio = n_cp/c.size
        if cp_ratio > 1e-6: 
            bar_cp = ax.bar(
                x=POSITIONS[key],
                height=cp_ratio,
                bottom=0,
                color=COLORS[key],
                label=rf"preventive",
            )
        if n_cp != c.size:
            bar_cc = ax.bar(
                x=POSITIONS[key],
                height=1.0-cp_ratio,
                bottom=cp_ratio,
                color=COLORS[key],
                label=rf"corrective",
            )
            for bar in bar_cc:
                bar.set_hatch("xxx")

    xticks = [POSITIONS[k] for k in results.keys()]
    xticklabels = [LABELS[k] for k in results.keys()]

    ax.set_xticks(xticks, xticklabels)
    ax.set_xlabel("Policy", labelpad=15)
    ax.set_ylabel("Preventive vs. corrective reps", rotation=90, labelpad=15)

    yticks = np.linspace(0, 1, 6)
    ax.set_yticks(yticks, yticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 0.99))
    ax.grid()
    plt.show()
    return