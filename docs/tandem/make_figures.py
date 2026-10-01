#!/usr/bin/env python3
"""Figures of the Tandem README (matplotlib, white background).

Numbers: the evaluation on a OnePlus 13T (Snapdragon 8 Elite) and a OnePlus 15 (Snapdragon 8 Elite Gen 5).
usage: python docs/tandem/make_figures.py      writes docs/tandem/step-time.svg and docs/tandem/time-saved.svg
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = "#2A78D6", "#EB6834"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "path", "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8, "xtick.color": "#444444", "ytick.color": "#222222", "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "svg.hashsalt": "tandem",
})

# server time per agent step (s): text step, screenshot step, as measured (bars and ratios) and as printed
ENGINES = [("Tandem (NPU)", (1.492, "1.49"), (4.87, "4.87")),
           ("llama.cpp (NPU)", (1.971, "1.97"), (5.80, "5.80")),
           ("llama.cpp (NPU, MTP)", (2.555, "2.56"), (7.60, "7.60")),
           ("MNN (GPU)", (4.98, "4.98"), (19.14, "19.1")),
           ("llama.cpp (GPU)", (5.075, "5.08"), (22.82, "22.8")),
           ("llama.cpp (CPU)", (11.19, "11.2"), (69.77, "69.8"))]

# server time per text step (s): Tandem, llama.cpp's NPU backend (its fastest configuration)
REDUCTIONS = [("Qwen3.5-2B", 0.767, 0.972), ("Qwen3.5-4B", 1.3515, 1.810), ("Qwen3.5-9B", 2.145, 3.0355),
              ("Agent policy, Snapdragon 8 Elite", 1.492, 1.971),
              ("Agent policy, Snapdragon 8 Elite Gen 5", 1.279, 1.6845),
              ("AndroidControl benchmark", 1.212, 1.580)]


def ratio(r):
    return f"{r:.2f}" if r < 2 else f"{r:.1f}" if r < 10 else f"{r:.0f}"


def step_time(path):
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    n, h = len(ENGINES), 0.36
    for i, (name, (t, ts), (s, ss)) in enumerate(ENGINES):
        y = i
        ax.barh(y - h / 2, t, height=h, color=BLUE, label="text step" if i == 0 else None)
        ax.barh(y + h / 2, s, height=h, color=ORANGE, label="screenshot step" if i == 0 else None)
        for v, shown, base, dy in ((t, ts, ENGINES[0][1][0], -h / 2), (s, ss, ENGINES[0][2][0], h / 2)):
            label = f"{shown} s" if i == 0 else f"{shown} s ({ratio(v / base)}×)"
            ax.text(v * 1.06, y + dy, label, va="center", fontsize=8.5, color="#222222")
    ax.set_xscale("log")
    ax.set_xlim(1, 150)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_yticks(range(n))
    ax.set_yticklabels([e[0] for e in ENGINES])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("server time per agent step (s, log scale); in parentheses: time relative to Tandem", fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def time_saved(path):
    fig, ax = plt.subplots(figsize=(8.0, 2.6))
    ys = [0, 1, 2, 3.4, 4.4, 5.4]  # a gap between the model sizes and the devices / workloads
    for y, (name, t, u) in zip(ys, REDUCTIONS):
        v = 100 * (1 - t / u)
        ax.barh(y, v, height=0.6, color=BLUE)
        ax.text(v + 0.4, y, f"{v:.0f}%", va="center", fontsize=8.5, color="#222222")
    ax.set_xlim(0, 35)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels(["0%", "10%", "20%", "30%"])
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in REDUCTIONS])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("less time per text step than llama.cpp on the NPU (its fastest configuration)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    for name, fn in (("step-time", step_time), ("time-saved", time_saved)):
        path = os.path.join(OUT, f"{name}.svg")
        fn(path)
        print("wrote", path)
