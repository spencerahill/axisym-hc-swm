"""Plot berimbau's parallel scaling for the SS09 model.

Data measured 2026-07-28 on berimbau (2x Xeon E5-2620 v4, 16 physical cores /
32 threads) with scripts/berimbau_timing.py, config ny=801 dt=30 ndays=200
backend=numba, machine otherwise idle (load average <1 at launch).

Panel (a): aggregate throughput vs concurrency, against ideal linear scaling.
Panel (b): per-run cost, i.e. the latency penalty each run pays for sharing
the machine. Together these set the worker cap: (a) says how much total work
gets done, (b) says what a single run gives up to get it.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = (
    "/tmp/claude-2296/-home-shill-py-axisym-hc-swm/"
    "25a58fbc-d4b1-4d27-8662-2e433bbe50cf/scratchpad/berimbau_scaling.png"
)

WORKERS = np.array([1, 2, 4, 8, 12, 16, 20, 24, 32])
S_PER_DAY = np.array(
    [0.0862, 0.0961, 0.0987, 0.1037, 0.1054, 0.1156, 0.1308, 0.1439, 0.1688]
)
THROUGHPUT = np.array(
    [11.60, 20.82, 40.55, 77.16, 113.86, 138.46, 152.85, 166.84, 189.61]
)

N_PHYS = 16  # physical cores

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
c_meas, c_ideal, c_phys = "#1f5fa8", "#999999", "#c1440e"

# --- (a) throughput -----------------------------------------------------
ax = axes[0]
ax.plot(WORKERS, THROUGHPUT[0] * WORKERS, "--", color=c_ideal, lw=1.4)
ax.plot(WORKERS, THROUGHPUT, "o-", color=c_meas, lw=2, ms=5)
ax.axvline(N_PHYS, color=c_phys, lw=1.2, ls=":")

ax.text(20.5, 300, "ideal linear", color=c_ideal, fontsize=9, rotation=38)
ax.text(21, 120, "measured", color=c_meas, fontsize=9, fontweight="bold")
ax.text(15.2, 20, "16 physical cores", color=c_phys, fontsize=8.5,
        rotation=90, va="bottom", ha="right")

ax.set_xlabel("concurrent runs")
ax.set_ylabel("aggregate throughput (model days / s)")
ax.set_title("(a) Total work done rises past 16, with hyperthreading",
             fontsize=10.5, loc="left")
ax.set_xlim(0, 34)
ax.set_ylim(0, 400)

# --- (b) per-run cost ---------------------------------------------------
ax = axes[1]
ax.plot(WORKERS, S_PER_DAY, "o-", color=c_meas, lw=2, ms=5)
ax.axhline(S_PER_DAY[0], color=c_ideal, lw=1.2, ls="--")
ax.axvline(N_PHYS, color=c_phys, lw=1.2, ls=":")

ax.text(2.0, 0.0885, "solo rate, 0.086 s/day", color=c_ideal, fontsize=9)
ax.text(24.5, 0.150, "measured", color=c_meas, fontsize=9,
        fontweight="bold", ha="center")
ax.text(15.2, 0.163, "16 physical cores", color=c_phys, fontsize=8.5,
        rotation=90, va="top", ha="right")
ax.annotate(
    "+34% per-run\nat 16 workers",
    xy=(16, 0.1156), xytext=(6.5, 0.132),
    fontsize=8.5, color="black",
    arrowprops=dict(arrowstyle="->", lw=1, color="black"),
)

ax.set_xlabel("concurrent runs")
ax.set_ylabel("per-run cost (s / model day)")
ax.set_title("(b) Each run's latency penalty for sharing the machine",
             fontsize=10.5, loc="left")
ax.set_xlim(0, 34)
ax.set_ylim(0.07, 0.18)

fig.suptitle(
    "berimbau parallel scaling, SS09 dry ny=801 dt=30 numba (measured 2026-07-28)",
    fontsize=11.5,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
