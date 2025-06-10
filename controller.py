import numpy as np
import matplotlib.pyplot as plt
from localPlanner import LocalOptimalControlPlanner, DifferentialDriveModel, BicycleModel
import time 

def wrap_angle(a):
    """Map any angle to (-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

def pose_error(actual, desired):
    """
    Return (theta_e, x_e, y_e) as in Eq. (13.30),
    with inputs ordered (x, y, theta).
    """
    x, y, theta   = actual
    x_d, y_d, theta_d = desired

    dtheta = wrap_angle(theta - theta_d)
    dx, dy = x - x_d, y - y_d

    c, s = np.cos(theta_d), np.sin(theta_d)
    x_e =  c*dx + s*dy
    y_e = -s*dx + c*dy
    return dtheta, x_e, y_e

def tracking_controller(actual_pose, desired_pose, desired_ctrl,
                        gains=(2.0, 6.0, 1.0), eps=1e-6):
    """
    Non-linear trajectory-tracking controller of Modern Robotics Eq. (13.31).

    Parameters
    ----------
    actual_pose  : (x, y, theta)
    desired_pose : (x_d, y_d, theta_d)
    desired_ctrl : (v_d,  omega_d)
    gains        : (k1, k2, k3)
    returns      : (v, omega)
    """
    k1, k2, k3 = gains
    theta_e, x_e, y_e = pose_error(actual_pose, desired_pose)

    # guard against division by zero when cos(theta_e) ~ 0
    if abs(np.cos(theta_e)) < eps:
        theta_e = np.sign(theta_e)*(np.pi/2 - eps)

    v_d, w_d = desired_ctrl
    v  = (v_d - k1*abs(v_d)*(x_e + y_e*np.tan(theta_e))) / np.cos(theta_e)
    w  = w_d - (k2*v_d*y_e + k3*abs(v_d)*np.tan(theta_e))*np.cos(theta_e)**2
    return v, w


def discrete_dynamics(state, control_input, dt, model_mismatch=False):
    """
    Update the robot's state based on its dynamics.

    Parameters:
    - state: Current state [x, y, theta]
    - control_input: Control input [linear_velocity, angular_velocity]
    - dt: Time step

    Returns:
    - Updated state [x, y, theta]
    """
    x, y, theta = state
    linear_velocity, angular_velocity = control_input
    if model_mismatch:
        # Introduce model mismatch by adding noise to the control input
        linear_velocity += np.random.normal(0, 1.0)
        angular_velocity += np.random.normal(0, 1.0)

    # Update state using differential drive kinematics
    x += linear_velocity * np.cos(theta) * dt
    y += linear_velocity * np.sin(theta) * dt
    theta += angular_velocity * dt

    # Normalize theta to keep it within [-pi, pi]
    theta = np.arctan2(np.sin(theta), np.cos(theta))

    return np.array([x, y, theta])


# ────────────────────────────────────────────────────────────────
#  Bicycle-model controller
# ────────────────────────────────────────────────────────────────
def tracking_controller_bicycle(actual_state, desired_state, desired_ctrl,
                                gains=(1.5, 3.0, 4.0), L=0.30):
    """
    actual_state  : [x, y, theta, phi]
    desired_state : [x_d, y_d, theta_d, phi_d]
    desired_ctrl  : [v_d, phi_dot_d]
    gains = (k1, k2, k3)
    """
    k1, k2, k3 = gains
    x,  y,  th,  phi  = actual_state
    xd, yd, thd, phid = desired_state
    vd, phi_dot_d     = desired_ctrl

    # pose error in desired frame
    dx, dy = x - xd, y - yd
    ex  =  np.cos(thd) * dx + np.sin(thd) * dy
    ey  = -np.sin(thd) * dx + np.cos(thd) * dy
    eth = (th  - thd + np.pi) % (2*np.pi) - np.pi
    eph = (phi - phid + np.pi) % (2*np.pi) - np.pi

    v_cmd     = vd * np.cos(eth) - k1 * ex
    phi_dot_c = phi_dot_d - k2 * np.sign(vd) * ey - k3 * eph
    return v_cmd, phi_dot_c 



# ────────────────────────────────────────────────────────────────
#  Euler propagation of the bicycle model
# ────────────────────────────────────────────────────────────────
def discrete_dynamics_bicycle(state, control_input, dt,
                              model_mismatch=False, L=0.30):
    """
    state  = [x, y, theta, phi]
    input  = [v, phi_dot]
    """
    x, y, th, phi = state
    v, phi_dot    = control_input
    if model_mismatch:
        v       += np.random.normal(0, 0.05*abs(v))
        phi_dot += np.random.normal(0, 0.05*abs(phi_dot))

    x   += v * np.cos(th) * dt
    y   += v * np.sin(th) * dt
    th  += (v * np.tan(phi) / L) * dt
    phi += phi_dot * dt
    th   = np.arctan2(np.sin(th), np.cos(th))       # wrap to -pi, pi
    return np.array([x, y, th, phi])



# ────────────────────────────────────────────────────────────────
#  1. helper – pose error for 3-state OR 4-state vectors
# ────────────────────────────────────────────────────────────────
def pose_error_generic(actual, desired):
    """
    Compute body-frame errors.  Works for
        • diff-drive: actual=(x,y,θ) ,  desired=(x_d,y_d,θ_d)
        • bicycle   : actual=(x,y,θ,φ), desired=(x_d,y_d,θ_d,φ_d)
    Returns a tuple: (e_θ, e_x, e_y, [e_φ])
    """
    x, y, theta, *rest  = actual
    xd, yd, thetad, *dr = desired

    ex_world  = x - xd
    ey_world  = y - yd
    c, s      = np.cos(thetad), np.sin(thetad)
    e_x       =  c*ex_world + s*ey_world
    e_y       = -s*ex_world + c*ey_world
    e_theta   = wrap_angle(theta - thetad)

    if rest and dr:                         # bicycle ⇒ steering error
        phi, phid = rest[0], dr[0]
        e_phi = wrap_angle(phi - phid)
        return e_theta, e_x, e_y, e_phi
    return e_theta, e_x, e_y


# ────────────────────────────────────────────────────────────────
# 2. run_tracking_test  (unified)
# ────────────────────────────────────────────────────────────────
def run_tracking_test(x_ref, u_ref, dt,
                      gains=(2.0, 6.0, 1.0),
                      noise=True,
                      is_bike=False):
    """
    Works for both models.
      • x_ref : (N,3) or (N,4)
      • u_ref : (N,2)
    """
    # choose proper controller & dynamics --------------------------------
    if is_bike:
        ctrl_fun = tracking_controller_bicycle
        dyn_fun  = discrete_dynamics_bicycle
        state_dim = 4
    else:
        ctrl_fun = tracking_controller
        dyn_fun  = discrete_dynamics
        state_dim = 3

    N = len(x_ref)

    # ----- logs ----------------------------------------------------------
    state_log  = np.empty((N, state_dim))
    cmd_log    = np.empty((N-1, 2))
    err_log    = np.empty((N, 4))   # (e_x,e_y,e_theta,e_phi)  (phi=0 for diff)

    # ----- initial state -------------------------------------------------
    x = x_ref[0].copy()
    x[0] += 0.05                     # small offsets
    x[1] += 0.05
    x[2] += np.deg2rad(5)
    state_log[0] = x

    err_log[0, :3] = 0.0
    if is_bike:
        err_log[0, 3] = 0.0

    # ----- main loop -----------------------------------------------------
    for k in range(N-1):
        des_state = x_ref[k]
        des_ctrl  = u_ref[k]

        # feedback command
        v_cmd, w_cmd = ctrl_fun(x, des_state, des_ctrl, gains=gains)

        # propagate “true” robot
        x = dyn_fun(x, (v_cmd, w_cmd), dt, model_mismatch=noise)


        # log
        state_log[k+1] = x
        cmd_log[k]     = (v_cmd, w_cmd)

        # errors
        errs = pose_error_generic(x, des_state)
        err_log[k+1, :len(errs)] = errs

    return state_log, cmd_log, err_log


# ────────────────────────────────────────────────────────────────
# 3. plotting (now 4 channels if bicycle)
# ────────────────────────────────────────────────────────────────
def plot_errors(t_ref, err_log, is_bike=False, gains = None):
    """
    err_log columns: 0=e_x,1=e_y,2=e_theta,3=e_phi (if bike)
    """
    n_rows = 4 if is_bike else 3
    fig, ax = plt.subplots(n_rows, 1, figsize=(7, 1.8*n_rows), sharex=True)

    lbls = [r'$x_e\,[m]$', r'$y_e\,[m]$', r'$\phi_e\,[rad]$', r'$\psi_e\,[rad]$']
    for i in range(n_rows):
        ax[i].plot(t_ref, err_log[:, i])
        ax[i].set_ylabel(lbls[i])
        ax[i].grid(True)
    ax[-1].set_xlabel('time [s]')
    fig.suptitle(f'{"Diff Drive" if not is_bike else "Bicycle"}: Tracking errors components -- k1,k2,k3 = ({",".join(map(str,gains))})')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__": 
    is_bike = True
    start = np.array([0.0, 0.0, 0.0])
    goal  = np.array([2.0, 2.0, 10 * np.pi / 180])  
    if is_bike: 
        start = np.array([0.0, 0.0, 0.0, 0.0])
        goal  = np.array([2.0, 2.0, 10 * np.pi / 180, 10 * np.pi / 180])  # [x, y, theta, phi]

    planner = LocalOptimalControlPlanner(
        model = DifferentialDriveModel() if not is_bike else BicycleModel(),
        t_final = 10.0,
        n_intervals = 500,
    )
    planner.build_ocp(start, goal)

    t_lin = np.linspace(0, 1, planner.n_intervals + 1) 
    x_guess = np.outer(1 - t_lin, start) + np.outer(t_lin, goal) # (n_intervals, 3)
    planner.set_initial_guess(x_guess=x_guess.T) 
    planner.solve()
    # ---- run planner (same as before) -----------------------------------
    t_ref, X_ref, U_ref = planner.sample()   # X_ref shape (N,4) for bicycle
    dt = t_ref[1] - t_ref[0]

    # gains = (0.2, 2., 7.)
    gains = (2.0, 6.0, 1.0) if not is_bike else (.2, 2., 4.5)
    # ---- simulate -------------------------------------------------------
    state_log, cmd_log, err_log = run_tracking_test(
            X_ref, U_ref, dt,
            gains=gains,
            noise=False,
            is_bike=is_bike
    )

    # ---- plot -----------------------------------------------------------
    plot_errors(t_ref, err_log, is_bike=is_bike, gains = gains)
