"""When does the drift in each suite-relevant quantity halt? Settling
analysis of the tier-0 1a/1b day-960->6000 extension records, plus the
two-epoch local-Ro profiles behind the 2026-07-17 cell-mean-Ro convention
change (memo run record). Run from the repo root; figures are written
beside the run outputs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ss09.read_output import load_centered

BASE = "model_output/fixed_ro_suite/tier0_amc_edge/"
RUNS = {
    "1a": (BASE + "amc_ss09_ny801_dt30_vd0_day6000.nc",
           BASE + "amc_ss09_ny801_dt30_vd0.nc"),
    "1b": (BASE + "amc_ss09_ny801_dt30_vd0_novert_day6000.nc",
           BASE + "amc_ss09_ny801_dt30_vd0_novert.nc"),
}
TAU_DRAG = 1157.4  # days, 1/(epsilon_u * 86400) at the default drag
T_E0 = 330.0 / 1.6


def trailing_mean(arr: np.ndarray, n: int, stride: int) -> tuple:
    """Trailing n-sample means of arr (time first axis) at given stride.
    Returns (end_indices, means)."""
    ends = np.arange(n, arr.shape[0] + 1, stride)
    means = np.stack([arr[e - n:e].mean(axis=0) for e in ends])
    return ends - 1, means


def settle_day(days: np.ndarray, dev: np.ndarray, eps: float) -> float:
    """Last day |dev| >= eps (dev in fractional units)."""
    bad = np.abs(dev) >= eps
    return float(days[bad][-1]) if bad.any() else float(days[0])


results = {}
for tag, (ext_path, orig_path) in RUNS.items():
    y, u_t, v_t, temp_t = load_centered(ext_path)  # (time, y) daily
    ds = xr.open_dataset(ext_path, decode_timedelta=False)
    jet = ds["north_jet_lat"].values
    ro_loc_t = ds["rossby_number"].values
    ds.close()
    ndays = jet.size
    beta = 2e-11
    nh = y > 0

    u_fin = u_t[-30:].mean(axis=0)
    ro_fin = ro_loc_t[-30:].mean(axis=0)
    jet_fin = jet[-30:].mean()
    core = nh & (y <= jet_fin) & np.isfinite(ro_fin)
    y_romax_fin = y[core][np.nanargmax(ro_fin[core])]
    band = core & (y >= y_romax_fin)
    basis = beta * y[band] ** 2 / 2

    ends, u_win = trailing_mean(u_t, 30, 10)
    _, v_win = trailing_mean(v_t, 30, 10)
    _, t_win = trailing_mean(temp_t, 30, 10)
    _, jet_win = trailing_mean(jet, 30, 10)
    wdays = 960 + ends
    ro_fit_series = np.array(
        [float(np.sum(uw[band] * basis) / np.sum(basis**2)) for uw in u_win]
    )
    depr_series = T_E0 - t_win[:, np.argmin(np.abs(y))]
    vmax_series = np.array([vw[nh].max() for vw in v_win])

    series = {
        "jet lat": (jet_win, jet_fin),
        "fitted Ro": (ro_fit_series, ro_fit_series[-1]),
        "depression": (depr_series, depr_series[-1]),
        "v_max": (vmax_series, vmax_series[-1]),
    }
    devs, settles = {}, {}
    for name, (s, ref) in series.items():
        dev = (s - ref) / ref
        devs[name] = dev
        settles[name] = {eps: settle_day(wdays, dev, eps)
                         for eps in (1e-3, 2e-4)}

    dev_jet = np.abs(devs["jet lat"])
    m = (wdays >= 1500) & (wdays <= 4000) & (dev_jet > 0)
    tau_eff = -1.0 / np.polyfit(wdays[m], np.log(dev_jet[m]), 1)[0]

    # Two-epoch local-Ro profiles and the (retired) argmax anchor.
    ds0 = xr.open_dataset(orig_path, decode_timedelta=False)
    ro_960 = ds0["rossby_number"].isel(time=slice(-30, None)).mean("time").values
    jet_960 = float(ds0["north_jet_lat"].isel(time=slice(-30, None)).mean())
    ds0.close()

    def y_romax_of(ro_prof, jet_pos):
        c = nh & (y <= jet_pos) & np.isfinite(ro_prof)
        return y[c][np.nanargmax(ro_prof[c])]

    results[tag] = dict(y=y, wdays=wdays, devs=devs, settles=settles,
                        tau_eff=tau_eff, ro_960=ro_960, ro_fin=ro_fin,
                        jet_fin=jet_fin,
                        anchors={"day960": y_romax_of(ro_960, jet_960),
                                 "day6000": y_romax_of(ro_fin, jet_fin)})

    print(f"\n=== {tag} ===")
    for name, s in settles.items():
        print(f"  {name:11s} settled to <0.1% of final by day "
              f"{s[1e-3]:.0f}; to <0.02% by day {s[2e-4]:.0f}")
    print(f"  jet-tail effective timescale: {tau_eff:.0f} d "
          f"(drag timescale {TAU_DRAG:.0f} d)")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for ax, tag in zip(axs, results):
    r = results[tag]
    for name, dev in r["devs"].items():
        ax.semilogy(r["wdays"], np.abs(dev) * 100, label=name, lw=1.6)
    ax.axhline(0.1, color="gray", lw=0.8, ls="--")
    ax.annotate("0.1%", (6000, 0.1), fontsize=8, color="gray",
                ha="right", va="bottom")
    ax.set_title(f"run {tag}: |deviation from final value|", fontsize=10)
    ax.set_xlabel("day")
    ax.set_ylim(1e-4, 20)
    ax.grid(alpha=0.25, lw=0.5)
axs[0].set_ylabel("|q(t) − q_final| / q_final  [%]")
axs[0].legend(fontsize=8, frameon=False)
fig.suptitle("Drift settling, 30-d trailing means (extensions day 960→6000)",
             fontsize=11)
fig.savefig(BASE + "settling_curves.png", dpi=150)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for ax, tag in zip(axs, results):
    r = results[tag]
    y6 = r["y"] / 1e6
    ax.plot(y6, r["ro_960"], color="tab:orange", lw=1.8,
            label="day 931-960 mean")
    ax.plot(y6, r["ro_fin"], color="black", lw=1.8, label="day 5971-6000 mean")
    for label, color in (("day960", "tab:orange"), ("day6000", "black")):
        ax.axvline(r["anchors"][label] / 1e6, color=color, lw=0.9, ls=":")
    ax.axvline(r["jet_fin"] / 1e6, color="gray", lw=0.9)
    ax.set_title(f"run {tag}: local Ro(y); dotted = retired y_Romax "
                 "anchors, gray = jet", fontsize=10)
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("y [10⁶ m]")
    ax.grid(alpha=0.25, lw=0.5)
axs[0].set_ylabel("local Ro")
axs[0].legend(fontsize=8, frameon=False)
fig.suptitle("Local Ro(y) at both epochs (the argmax anchor jump)",
             fontsize=11)
fig.savefig(BASE + "ro_profiles_epochs.png", dpi=150)
plt.close(fig)
print("\nfigures:", BASE + "settling_curves.png,",
      BASE + "ro_profiles_epochs.png")
