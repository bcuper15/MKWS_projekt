"""
Ignition delay study for CH4/H2/air mixtures using a 0D constant-volume,
adiabatic reactor (ct.IdealGasReactor + ct.ReactorNet).

The ignition delay time tau is defined as the time at which dT/dt is
maximal (steepest temperature rise = main ignition event).

Computes:
  - tau vs initial temperature T0, for several X_H2, at fixed P0
    (plotted as an Arrhenius-type plot: log10(tau) vs 1000/T0)
  - tau vs initial pressure P0, for X_H2 = 0.0, 0.5, 1.0, at fixed T0
  - tau vs X_H2 (cross-section of the T0 scan at one T0)

Outputs:
  - figures/09_ignition_delay_Arrhenius.png
  - figures/10_ignition_delay_vs_pressure.png
  - figures/11_ignition_delay_vs_xH2.png
  - data/ignition_delay_T0_scan.csv
  - data/ignition_delay_P0_scan.csv
"""

from pathlib import Path

import cantera as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mixtures import make_gas, set_mixture

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

PHI = 1.0
X_H2_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]
T0_LIST = [900, 950, 1000, 1050, 1100, 1150, 1200, 1300, 1400]  # K
P0_REF_BAR = 10.0

P_LIST_BAR = [1.0, 10.0, 25.0]
T0_REF = 1100.0  # K, used for the pressure scan and the X_H2 cross-section

T0_SCAN_CSV = DATA_DIR / "ignition_delay_T0_scan.csv"
P0_SCAN_CSV = DATA_DIR / "ignition_delay_P0_scan.csv"


def ignition_delay(T0: float, P0: float, phi: float, x_h2: float,
                    t_end: float = 20.0, max_steps: int = 100000) -> float:
    """Return ignition delay time [s], defined as time of max dT/dt.

    Returns np.nan if the mixture does not ignite (T rise < 400 K) within
    t_end seconds.
    """
    gas = make_gas()
    set_mixture(gas, phi, x_h2, T0, P0)
    reactor = ct.IdealGasReactor(gas, clone=False)
    sim = ct.ReactorNet([reactor])

    ts = [0.0]
    Ts = [reactor.T]
    for _ in range(max_steps):
        t = sim.step()
        ts.append(t)
        Ts.append(reactor.T)
        if t > t_end:
            break
        # Stop once the temperature has plateaued well above T0
        if reactor.T > T0 + 600 and (Ts[-1] - Ts[-2]) < 1e-3:
            break

    ts = np.array(ts)
    Ts = np.array(Ts)
    if Ts[-1] - Ts[0] < 400:
        return np.nan

    dTdt = np.gradient(Ts, ts)
    return ts[np.argmax(dTdt)]


def _load_cache(path: Path, key_cols) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in key_cols + ["tau"]})


def run_T0_scan() -> pd.DataFrame:
    df = _load_cache(T0_SCAN_CSV, ["x_h2", "T0"])
    done = set(zip(df.x_h2.round(6), df.T0.round(6)))

    for x_h2 in X_H2_LIST:
        for T0 in T0_LIST:
            key = (round(x_h2, 6), round(float(T0), 6))
            if key in done:
                continue
            tau = ignition_delay(T0, P0_REF_BAR * 1e5, PHI, x_h2)
            df = pd.concat([df, pd.DataFrame([{"x_h2": x_h2, "T0": T0, "tau": tau}])],
                            ignore_index=True)
            done.add(key)
    df.to_csv(T0_SCAN_CSV, index=False)
    return df


def run_P0_scan() -> pd.DataFrame:
    df = _load_cache(P0_SCAN_CSV, ["x_h2", "P0_bar"])
    done = set(zip(df.x_h2.round(6), df.P0_bar.round(6)))

    for x_h2 in [0.0, 0.5, 1.0]:
        for P_bar in P_LIST_BAR:
            key = (round(x_h2, 6), round(P_bar, 6))
            if key in done:
                continue
            tau = ignition_delay(T0_REF, P_bar * 1e5, PHI, x_h2)
            df = pd.concat([df, pd.DataFrame([{"x_h2": x_h2, "P0_bar": P_bar, "tau": tau}])],
                            ignore_index=True)
            done.add(key)
    df.to_csv(P0_SCAN_CSV, index=False)
    return df


def plot_arrhenius(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for x_h2 in X_H2_LIST:
        d = df[df.x_h2 == x_h2].sort_values("T0")
        ax.semilogy(1000.0 / d.T0, d.tau * 1000, "o-", linewidth=2, label=f"X_H2 = {x_h2:.2f}")
    ax.set_xlabel("1000 / T0 [1/K]")
    ax.set_ylabel("Ignition delay, tau [ms]")
    ax.set_title(f"Ignition delay vs initial temperature (Arrhenius plot)\n"
                  f"CH4/H2/air, phi={PHI}, p0={P0_REF_BAR:.0f} bar, constant-volume reactor")
    ax.legend(title="Fuel blend", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_ignition_delay_Arrhenius.png", dpi=150)
    plt.close(fig)


def plot_vs_pressure(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for x_h2 in [0.0, 0.5, 1.0]:
        d = df[df.x_h2 == x_h2].sort_values("P0_bar")
        ax.loglog(d.P0_bar, d.tau * 1000, "o-", linewidth=2, label=f"X_H2 = {x_h2:.2f}")
    ax.set_xlabel("Initial pressure, p0 [bar]")
    ax.set_ylabel("Ignition delay, tau [ms]")
    ax.set_title(f"Ignition delay vs pressure\nCH4/H2/air, phi={PHI}, T0={T0_REF:.0f} K")
    ax.legend(title="Fuel blend", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_ignition_delay_vs_pressure.png", dpi=150)
    plt.close(fig)


def plot_vs_xh2(df_T0: pd.DataFrame, T0_ref: float = T0_REF):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    d = df_T0[np.isclose(df_T0.T0, T0_ref)].sort_values("x_h2")
    ax.semilogy(d.x_h2, d.tau * 1000, "o-", linewidth=2, color="darkgreen")
    ax.set_xlabel("H2 fraction in fuel, X_H2 [-]")
    ax.set_ylabel("Ignition delay, tau [ms]")
    ax.set_title(f"Ignition delay vs H2 blend fraction\n"
                  f"CH4/H2/air, phi={PHI}, T0={T0_ref:.0f} K, p0={P0_REF_BAR:.0f} bar")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_ignition_delay_vs_xH2.png", dpi=150)
    plt.close(fig)


def run():
    FIG_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    print("Ignition delay study: T0 scan...")
    df_T0 = run_T0_scan()
    plot_arrhenius(df_T0)
    plot_vs_xh2(df_T0)

    print("Ignition delay study: P0 scan...")
    df_P0 = run_P0_scan()
    plot_vs_pressure(df_P0)

    print("\nDone. Figures saved to:", FIG_DIR)


if __name__ == "__main__":
    run()
