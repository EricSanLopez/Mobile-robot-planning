from __future__ import annotations

import argparse
import datetime as dtt
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

# ────────────────────────────────────────────────────────────────────────────────
# EKF core
# ────────────────────────────────────────────────────────────────────────────────
class EKF:
    """Minimal Extended Kalman Filter."""

    def __init__(
        self,
        x0_hat: np.ndarray,
        P0: np.ndarray,
        f_fun,
        F_fun,
        Q: np.ndarray,
    ) -> None:
        self.x_hat = x0_hat.copy()
        self.P = P0.copy()
        self.f_fun = f_fun
        self.F_fun = F_fun
        self.Q = Q.copy()

    # ── prediction ──────────────────────────────────────────────────────────
    def predict(self, u: np.ndarray, dt: float) -> None:
        self.x_hat = self.f_fun(self.x_hat, u, dt)
        Fk = self.F_fun(self.x_hat, u, dt)
        self.P = Fk @ self.P @ Fk.T + self.Q  

    # ── update ─────────────────────────────────────────────────────────────
    def update(self, z: np.ndarray, h_fun, H_fun, R: np.ndarray) -> float:
        nu = z - h_fun(self.x_hat)                     # innovation
        Hk = H_fun(self.x_hat)
        S = Hk @ self.P @ Hk.T + R                     # innovation covariance
        K = self.P @ Hk.T @ np.linalg.inv(S)           # Kalman gain
        self.x_hat += K @ nu 
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ Hk) @ self.P         
        # self.P = (I - K @ Hk) @ self.P @ (I - K @ Hk).T + K @ R @ K.T
        return float(nu.T @ np.linalg.inv(S) @ nu)     # NIS (scalar)


# ────────────────────────────────────────────────────────────────────────────────
# Motion & measurement models
# ────────────────────────────────────────────────────────────────────────────────

def f_truth(x: np.ndarray, u: np.ndarray, dt: float, *, sigma_v: float, sigma_w: float) -> np.ndarray:
    """Ground‑truth propagation with *control* noise (ṽ, ω̃)."""
    v_noisy = u[0] + np.random.normal(0.0, sigma_v)
    w_noisy = u[1] + np.random.normal(0.0, sigma_w)
    th = x[2]
    return x + np.array([
        v_noisy * np.cos(th) * dt,
        v_noisy * np.sin(th) * dt,
        w_noisy * dt,
    ])


def f_est(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Estimator's (noise‑free) motion model."""
    th = x[2]
    return x + np.array([
        u[0] * np.cos(th) * dt,
        u[0] * np.sin(th) * dt,
        u[1] * dt,
    ])


def F_est(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Exact Jacobian ∂f/∂x (compact vectorised form)."""
    th = x[2]
    v = u[0]
    return np.array([
        [1.0, 0.0, -v * dt * np.sin(th)],
        [0.0, 1.0,  v * dt * np.cos(th)],
        [0.0, 0.0,  1.0],
    ])

# ────────────────────────────────────────────────────────────────────────────────
# Bicycle model dynamics: 
# ────────────────────────────────────────────────────────────────────────────────

# ─── bicycle truth propagation (with noisy controls) ─────────────
def f_truth_bike(x: np.ndarray, u: np.ndarray, dt: float, *,
            sigma_v: float, sigma_w: float, L: float = 0.50) -> np.ndarray:
    v_noisy   = u[0] + np.random.normal(0.0, sigma_v)
    phi_dot_n = u[1] + np.random.normal(0.0, sigma_w)
    th, phi   = x[2], x[3]
    return x + np.array([
        v_noisy * np.cos(th) * dt,
        v_noisy * np.sin(th) * dt,
        v_noisy * np.tan(phi)/L * dt,
        phi_dot_n * dt,
    ])

# ─── EKF internal model (noise-free) ─────────────────────────────
def f_est_bike(x: np.ndarray, u: np.ndarray, dt: float, L: float = 0.50) -> np.ndarray:
    th, phi = x[2], x[3]
    v, phi_dot = u
    return x + np.array([
        v * np.cos(th) * dt,
        v * np.sin(th) * dt,
        v * np.tan(phi)/L * dt,
        phi_dot * dt,
    ])

# ─── Jacobian ∂f/∂x ─────────────────────────────────────────────
def F_est_bike(x: np.ndarray, u: np.ndarray, dt: float, L: float = 0.50) -> np.ndarray:
    th, phi = x[2], x[3]
    v = u[0]
    sec2 = 1.0 / np.cos(phi)**2
    return np.array([
        [1.0, 0.0, -v * dt * np.sin(th),            0.0],
        [0.0, 1.0,  v * dt * np.cos(th),            0.0],
        [0.0, 0.0,  1.0,           v * dt * sec2 / L],
        [0.0, 0.0,  0.0,                         1.0],
    ])



def h_gps(x: np.ndarray) -> np.ndarray:
    return x[:2]


def H_gps(_: np.ndarray) -> np.ndarray:  # (2×3) constant Jacobian
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

def H_gps_bike(_: np.ndarray) -> np.ndarray:  # (2×4) constant Jacobian
    return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

def Q_from_controls(x: np.ndarray, dt: float, sigma_v: float, sigma_w: float) -> np.ndarray:
    """First‑order propagation of control noise to state space."""
    theta, *_ = x
    G = np.array([
        [np.cos(theta) * dt, 0.0],
        [np.sin(theta) * dt, 0.0],
        [0.0, dt],
    ])
    return G @ np.diag([sigma_v ** 2, sigma_w ** 2]) @ G.T

def Q_from_controls_bike(x: np.ndarray, dt, sigma_v, sigma_w, L=0.30): 
    theta,phi = x
    G = np.array([[np.cos(theta)*dt, .0],
     [np.sin(theta)*dt, .0],
     [np.tan(phi) / L * dt, .0],
     [0, dt]])
    
    return G @ np.diag([sigma_v ** 2, sigma_w ** 2]) @ G.T


# ────────────────────────────────────────────────────────────────────────────────
# Utility – nice 2‑σ ellipse plot
# ────────────────────────────────────────────────────────────────────────────────

def plot_cov_ellipse(mu: np.ndarray, P: np.ndarray, ax, *, n_sigma: int = 2, **kwargs):
    vals, vecs = np.linalg.eigh(P)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2 * n_sigma * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ax.add_patch(Ellipse(xy=mu, width=width, height=height, angle=angle, **kwargs))


# ────────────────────────────────────────────────────────────────────────────────
# Main experiment
# ────────────────────────────────────────────────────────────────────────────────

def run_mc(
    *,
    sigma_v_pct: float,
    sigma_w_pct: float,
    total_time: float,
    gps_latency: float,
    v: float,
    w: float,
    save_plots: bool,
    is_bike: bool = False,
):
    # ——— simulation constants ————————————————————————————————
    N_MC = 100
    dt = 0.01
    sensor_noise_std = 0.03  # [m]

    # controls (constant)
    u = np.array([v, w])  # [m/s, rad/s]
    sigma_v = abs(u[0]) * sigma_v_pct
    sigma_w = abs(u[1]) * sigma_w_pct

    T = int(np.ceil(total_time / dt))
    gps_every = int(np.round(gps_latency / dt))

    # initial truth / estimate
    x0 = np.array([0.0, 0.0] + [-np.deg2rad(30.0), 0.] )
    x0_hat = np.array([0.2, -0.1] + [-np.deg2rad(10.0), 0.]) 
    # initial uncertainty: 0.5 m in X/Y and 15 degs in angles
    diag = [.5**2] * 2 + [np.deg2rad(15.0)**2, np.deg2rad(5)**2]
    if not is_bike:
        x0 = x0[:3]  
        x0_hat = x0_hat[:3] 
        diag = diag[:3]  # 3-state
    P0 = np.diag(diag)

    # measurement noise
    R_gps = np.diag([sensor_noise_std ** 2, sensor_noise_std ** 2])

    # logs for consistency stats
    nis_all, nees_all = [], []

    # store trajectory of first run for pretty plots
    first_gt, first_hat, first_P = [], [], []

    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi

    for j in range(N_MC):
        rng = np.random.default_rng(j)
        x_t = x0.copy()
        f_func = f_est_bike if is_bike else f_est
        F_func = F_est_bike if is_bike else F_est
        f_true = f_truth_bike if is_bike else f_truth 
        Q_func = Q_from_controls_bike if is_bike else Q_from_controls
        ekf = EKF(x0_hat, P0, f_func, F_func, Q_func(x0[2:] if is_bike else x0[2:3], dt, sigma_v, sigma_w))

        for k in range(T):
            # —— propagate ground truth (with *noisy* controls) ————————
            x_t = f_true(x_t, u, dt, sigma_v=sigma_v, sigma_w=sigma_w)

            # —— EKF prediction ————————————————————————————————
            hat = ekf.x_hat
            ekf.Q = Q_func(hat[2:] if is_bike else hat[2:3], dt, sigma_v, sigma_w)
            ekf.predict(u, dt)

            # —— GPS update ——————————————————————————————————————
            if k % gps_every == 0:
                z = h_gps(x_t) + rng.normal(0.0, sensor_noise_std, 2)
                nis = ekf.update(z, h_gps, H_gps_bike if is_bike else H_gps, R_gps)
                nis_all.append(nis)

            # —— NEES for consistency ——————————————————————————
            err = ekf.x_hat - x_t
            nees_all.append(float(err.T @ np.linalg.inv(ekf.P) @ err))

            if j == 0:  # log first run
                first_gt.append(x_t.copy())
                first_hat.append(ekf.x_hat.copy())
                first_P.append(ekf.P.copy())

    # ——— plotting ————————————————————————————————————————————————
    timestamp = dtt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("plots")
    if save_plots:
        save_dir.mkdir(exist_ok=True)

    # ≈≈ NIS & NEES histograms ≈≈
    def _plot_chi(vals, df, title, fname):
        xs = np.linspace(0, chi2.ppf(0.999, df=df), 300)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(vals, 30, density=True, alpha=0.6)
        plt.plot(xs, chi2.pdf(xs, df), "k--")
        plt.title(f"{title} histogram")
        plt.xlabel(title);
        plt.ylabel("pdf")

        plt.subplot(1, 2, 2)
        emp_cdf = np.searchsorted(np.sort(vals), xs, side="right") / len(vals)
        plt.plot(xs, emp_cdf)
        plt.plot(xs, chi2.cdf(xs, df), "k--")
        plt.title(f"{title} CDF")
        plt.xlabel(title);
        plt.ylabel("cumulative probability")
        plt.tight_layout()
        if save_plots:
            plt.savefig(save_dir / f"{timestamp}_{fname}.png", dpi=150)
        plt.show()

    nis_arr = np.array(nis_all) 
    nees_arr = np.array(nees_all)
    nis_cov = (nis_arr <= chi2.ppf(0.95, df=2)).mean()
    nes_cov = (nees_arr <= chi2.ppf(0.95, df=3 if not is_bike else 4)).mean()

    print(f'NEES 95 % coverage = {100*nes_cov:.2f} %')
    print(f'NIS 95 % coverage = {100*nis_cov:.2f} %')

    _plot_chi(nis_arr, df=2, title="NIS", fname="nis")
    _plot_chi(nees_arr, df=3 if not is_bike else 4, title="NEES", fname="nees")


    # —— trajectory of first run ————————————————————————————————
    t_vec = np.arange(len(first_gt)) * dt
    x_gt = np.stack(first_gt)
    x_hat = np.stack(first_hat)
    P_hist = np.stack(first_P)

    max_int = 4 if is_bike else 3
    fig, ax = plt.subplots(max_int, 1, figsize=(10, 8), sharex=True)
    labels = [r"$X$ position [m]", r"$Y$ position [m]", r"Heading $\phi$ [rad]", r"Steering $\psi$ [rad]"]
    for i in range(max_int):
        x_gt_ = x_gt[:, i]
        x_hat_ = x_hat[:, i]
        if i == 2: 
            x_gt_ = wrap(x_gt_) # wrap angle to [-pi, pi]
            x_hat_ = wrap(x_hat_)
        ax[i].plot(t_vec, x_gt_, "--", label="ground truth")
        ax[i].plot(t_vec, x_hat_, "k", label="EKF estimate")
        sigma = np.sqrt(P_hist[:, i, i])
        ax[i].fill_between(t_vec, x_hat_ - 3 * sigma, x_hat_ + 3 * sigma, color="k", alpha=0.15)
        ax[i].set_ylabel(labels[i])
        ax[i].grid(True)
    ax[0].legend(ncol=2)
    ax[-1].set_xlabel("time [s]")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_plots:
        plt.savefig(save_dir / f"{timestamp}_state_trace.png", dpi=150)
    plt.show()

    
    fig, ax_err = plt.subplots(max_int, 1, figsize=(10, 8), sharex=True)
    coord_lbl = [
        r"$|\hat X - X|$ [m]",
        r"$|\hat Y - Y|$ [m]",
        r"$|\hat\phi - \phi|$ [rad]",
        r"$|\hat\psi - \psi|$ [rad]",
    ]

    for i, ax_ in enumerate(ax_err):
        x_gt_ = x_gt[:, i]
        x_hat_ = x_hat[:, i]
        err   = np.abs(x_gt_ - x_hat_)  # abs error
        if i == 2:
            err = wrap(err)  # wrap angle error to [-pi, pi]
        sigma = np.sqrt(P_hist[:, i, i])          # 1-σ from EKF covariance

        ax_.plot(t_vec, err, label='|error|')
        ax_.fill_between(t_vec, err-sigma, err+sigma,
                         color='k', alpha=0.15,
                         label=r'$\pm \sigma$' if i == 0 else "")
        # ax_.set_yscale('log')
        ax_.set_ylabel(coord_lbl[i])
        ax_.grid(True)

    ax_err[-1].set_xlabel('time [s]')
    ax_err[0].legend(ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_plots:
        plt.savefig(save_dir / f"{timestamp}_error_vs_sigma.png", dpi=150)

    plt.show()


    # —— XY path with covariance ellipses ————————————————————————
    fig, axe = plt.subplots(figsize=(6, 6))
    axe.plot(x_gt[:, 0], x_gt[:, 1], "g--", label="ground truth path")
    axe.plot(x_hat[:, 0], x_hat[:, 1], "k", label="EKF path")
    for i in range(0, len(t_vec), int(0.3 / dt)):
        plot_cov_ellipse(x_hat[i, :2], P_hist[i, :2, :2], axe, n_sigma=3, edgecolor="k", facecolor="k", alpha=0.2)
    axe.set_xlabel("X [m]")
    axe.set_ylabel("Y [m]")
    axe.set_aspect("equal", adjustable="box")
    axe.grid(True)
    axe.legend()
    if save_plots:
        plt.savefig(save_dir / f"{timestamp}_xy_cov.png", dpi=150)
    plt.show()


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EKF Monte‑Carlo consistency test (GPS only)")
    parser.add_argument("--sigma_v_pct", type=float, default=0.05, help="σ_v as fraction of commanded v (default 0.05)")
    parser.add_argument("--sigma_w_pct", type=float, default=0.05, help="σ_ω as fraction of commanded ω (default 0.05)")
    parser.add_argument("--total_time", type=float, default=8.0, help="Simulation length in seconds (default 8)")
    parser.add_argument("--gps_latency", type=float, default=0.2, help="GPS period in seconds (default 0.2 ≈ 5 Hz)")
    parser.add_argument('--vel', type=float, default=1.0, help='v mod (m/s)')
    parser.add_argument('--w', type=float, default=-1.0, help='ω mod (rad/s)')
    parser.add_argument("--save_plots", action="store_true", help="Save figures to ./plots directory")
    parser.add_argument('--bike_L', type=float, default=0.00, help='L > 0 uses the bike model rather than diff drive')


    args = parser.parse_args()
    run_mc(
        sigma_v_pct=args.sigma_v_pct,
        sigma_w_pct=args.sigma_w_pct,
        total_time=args.total_time,
        gps_latency=args.gps_latency,
        save_plots=args.save_plots,
        v = args.vel, 
        w = args.w,
        is_bike = args.bike_L > 0.0,
    )


if __name__ == "__main__":
    main()
