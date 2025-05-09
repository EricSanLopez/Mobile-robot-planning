from __future__ import annotations

import numpy as np
import casadi as ca
import rockit


class DifferentialDriveModel:
    """Symbolic differential‑drive dynamics

    State:  x = [x, y, θ]ᵀ (planar pose)  
    Control: u = [v, ω]ᵀ (linear & angular velocity)
    """

    nstate: int = 3
    ncontrol: int = 2

    @staticmethod
    def dynamics(x: ca.MX | ca.SX, u: ca.MX | ca.SX) -> ca.MX:
        """Returns the time‑derivative x_dot with shape (3,) as a CasADi symbolic expression.

        Parameters
        ----------
        x : (3,) casadi vector – current state [x, y, θ]
        u : (2,) casadi vector – control  [v, ω]
        """
        px, py, theta = ca.vertsplit(x)
        v, omega = ca.vertsplit(u)

        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = omega
        return ca.vcat([dx, dy, dtheta])


class LocalOptimalControlPlanner:

    def __init__(
        self,
        model: DifferentialDriveModel,
        t_final: float,
        n_intervals: int = 40,
        ipopt_options: dict | None = None,
    ) -> None:
        """Create an *empty* OCP that can later be populated via :meth:`build_ocp`.

        Parameters
        ----------
        model        : dynamics model providing a ``dynamics(x,u)`` method
        t_final      : horizon length (seconds)
        n_intervals  : # of control intervals for multiple‑shooting
        ipopt_options: optional dict of IPOPT options (verbosity, tolerances…)
        """
        self.model = model
        self.t_final = float(t_final)
        self.n_intervals = int(n_intervals)

        # Will be set by *build_ocp* -----------------------------------------
        self.ocp: rockit.Ocp | None = None
        self.x:   rockit.ForwardRef | None = None  # state variable handle
        self.u:   rockit.ForwardRef | None = None  # control variable handle
        self._sol: rockit.solution.Solution | None = None

        # Solver options ------------------------------------------------------
        default_ipopt = {
            "ipopt.print_level": 3,
            "print_time": True,
            "ipopt.max_iter": 500,
        }
        if ipopt_options:
            default_ipopt.update(ipopt_options)
        self.ipopt_options = default_ipopt

    # ---------------------------------------------------------------------
    # Problem construction
    # ---------------------------------------------------------------------

    def build_ocp(
        self,
        x_start: np.ndarray,
        x_goal: np.ndarray,
        state_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        control_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        if x_start.shape != (self.model.nstate,) or x_goal.shape != (
            self.model.nstate,
        ):
            raise ValueError("Start/goal state must have shape (3,)")

        ocp = rockit.Ocp(t0=0.0, T=self.t_final) # in seconds

        # ─── Decision variables ────────────────────────────────────────────
        x = ocp.state(self.model.nstate)
        u = ocp.control(self.model.ncontrol)

        # ─── Dynamics ─────────────────────────────────────
        ocp.set_der(state=x, der=self.model.dynamics(x, u))

        # ─── Boundary conditions ───────────────────────────────────────────
        ocp.subject_to(ocp.at_t0(x) == x_start)
        ocp.subject_to(ocp.at_tf(x) == x_goal)

        # ─── State & control bounds (optional) ─────────────────────────────
        if state_bounds is not None:
            x_min, x_max = state_bounds
            ocp.subject_to(x_min <= x)
            ocp.subject_to(x <= x_max)
        if control_bounds is not None:
            u_min, u_max = control_bounds
            ocp.subject_to(u_min <= u)
            ocp.subject_to(u <= u_max)

        # ─── Obstacle avoidance ─────────────────────────────────────
        # 
        # ─── Cost function ──────────────────────────────────────────
        # Typical choices include: ∫  ||u||² dt, ∫ 1 dt (minimum‑time), etc.
        
        ocp.add_objective(ocp.integral(ca.sumsqr(u)))
        

        # ─── Discretise & choose solver ────────────────────────────────────
        ocp.method(rockit.MultipleShooting(N=self.n_intervals))
        ocp.solver("ipopt", self.ipopt_options)

        self.ocp, self.x, self.u = ocp, x, u

    # ---------------------------------------------------------------------
    # Solving utilities
    # ---------------------------------------------------------------------

    def set_initial_guess(
        self,
        x_guess: np.ndarray | None = None,
        u_guess: np.ndarray | None = None,
    ) -> None:
        """(Optional) provide initial guesses for the trajectory e.g. linear """
        if self.ocp is None:
            raise RuntimeError("Call build_ocp() first.")
        if x_guess is not None:
            self.ocp.set_initial(self.x, x_guess) # initial guess != initial conditions
        if u_guess is not None:
            self.ocp.set_initial(self.u, u_guess)

    def solve(self) -> None:
        """Solve the OCP and store the solution internally."""
        if self.ocp is None:
            raise RuntimeError("Problem not built. Call build_ocp() first.")
        self._sol = self.ocp.solve()


    def sample(self, grid: str = "control") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (t, X, U) arrays sampled on the chosen grid ("control"|"integrator")."""
        if self._sol is None:
            raise RuntimeError("No solution available. Call solve() first.")
        t, X = self._sol.sample(self.x, grid=grid)
        _, U = self._sol.sample(self.u, grid=grid)
        return t, X, U

    # Convenience shorthand -------------------------------------------------
    def get_trajectory(self):
        """Alias for ``sample()[1]`` (state trajectory)."""
        return self.sample()[1]


if __name__ == "__main__":
    start = np.array([0.0, 0.0, 0.0])
    goal  = np.array([1.0, 1.0, np.pi / 2])

    planner = LocalOptimalControlPlanner(
        model = DifferentialDriveModel(),
        t_final = 5.0,
        n_intervals = 30,
    )
    planner.build_ocp(start, goal)

    # straight‑line initial guess in configuration space 
    t_lin = np.linspace(0, 1, planner.n_intervals + 1) 
    x_guess = np.outer(1 - t_lin, start) + np.outer(t_lin, goal) # (n_intervals, 3)
    planner.set_initial_guess(x_guess=x_guess.T)  # rockit expects column‑wise (3; n_intervals)

    planner.solve()
    t, X, U = planner.sample()

    print(f"Solved. Satate shape: {X.shape = }, {U.shape = }") # (n_intervals, 3)
    print("Solved.✅ Final state:", X[-1]) # (3,)
