import numpy as np
import matplotlib.pyplot as plt
from localPlanner import LocalOptimalControlPlanner, DifferentialDriveModel
import time 
from utils import show_robot

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


def run_tracking_test(x_ref,            # (N,3)  reference poses [x_d, y_d, theta_d]
                      u_ref,            # (N,2)  reference controls [v_d, w_d]
                      dt,
                      gains=(2.0, 6.0, 1.0),
                      noise=True,
                      visualize_robot=False
                      ):
    """
    Simulate the robot with the nonlinear tracking controller and
    return time-stamped logs of pose, controls and errors.
    """
    N               = len(x_ref)
    # pre-allocate logs
    x_log           = np.empty(N)
    y_log           = np.empty(N)
    theta_log       = np.empty(N)
    x_e_log         = np.empty(N)
    y_e_log         = np.empty(N)
    theta_e_log     = np.empty(N)
    v_cmd_log       = np.empty(N)
    w_cmd_log       = np.empty(N)

    # --- initial state --------------------------------------------------------
    x = x_ref[0,0] + 0.05
    y = x_ref[0,1] + 0.05
    theta = x_ref[0,2] + np.deg2rad(5)
  
    x_log[0], y_log[0], theta_log[0] = x, y, theta
    x_e_log[0] = y_e_log[0] = theta_e_log[0] = 0.0

    # --- main loop ------------------------------------------------------------
    for k in range(N-1):
        des_pose = tuple(x_ref[k])      # (x_d, y_d, theta_d)
        des_ctrl = tuple(u_ref[k])      # (v_d, w_d)

        # compute feedback command
        v_cmd, w_cmd = tracking_controller(
            (x, y, theta),
            des_pose,
            des_ctrl,
            gains=gains
        )

        
        # propagate the "true" robot
        x, y, theta = discrete_dynamics(
            (x, y, theta),
            (v_cmd, w_cmd),
            dt,
            model_mismatch=noise
        )
        if visualize_robot:
            show_robot(x, y, theta)
            time.sleep(dt)  # simulate real-time

        # log state and command
        x_log[k+1],    y_log[k+1],    theta_log[k+1] = x, y, theta
        v_cmd_log[k],  w_cmd_log[k] = v_cmd, w_cmd

        # compute and log tracking error
        theta_e, x_e, y_e = pose_error(
            (x, y, theta),
            des_pose
        )
        x_e_log[k+1], y_e_log[k+1], theta_e_log[k+1] = x_e, y_e, theta_e

    return (x_log, y_log, theta_log,
            x_e_log, y_e_log, theta_e_log,
            v_cmd_log, w_cmd_log)



if __name__ == "__main__": 
    start = np.array([0.0, 0.0, 0.0])
    goal  = np.array([2.0, 2.0, 10 * np.pi / 180])  

    planner = LocalOptimalControlPlanner(
        model = DifferentialDriveModel(),
        t_final = 10.0,
        n_intervals = 500,
    )
    planner.build_ocp(start, goal)

    t_lin = np.linspace(0, 1, planner.n_intervals + 1) 
    x_guess = np.outer(1 - t_lin, start) + np.outer(t_lin, goal) # (n_intervals, 3)
    planner.set_initial_guess(x_guess=x_guess.T) 
    planner.solve()
    t_ref, x_ref, u_ref = traj = planner.sample() # references 

    dt                  = t_ref[1] - t_ref[0]
    gains = (2.0, 2.0, 12.0)  # K1, K2, K3
    gains = (2.0, 6.0, 1.0)  # K1, K2, K3
    logs = run_tracking_test(x_ref, u_ref, dt,
                            gains=gains,
                            noise=False,
                            visualize_robot=False
                            )

    (x_log, y_log, theta_log,
    x_e, y_e, theta_e,
    v_cmd, w_cmd) = logs

    # ------------------------------------------------------------------
    # plot tracking errors
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    ax[0].plot(t_ref, theta_e, label=r'$\theta_e$')
    ax[0].set_ylabel(r'$\theta_e$ [rad]')
    ax[0].grid()

    ax[1].plot(t_ref, x_e, label=r'$x_e$')
    ax[1].set_ylabel(r'$x_e$ [m]')
    ax[1].grid()

    ax[2].plot(t_ref, y_e, label=r'$y_e$')
    ax[2].set_ylabel(r'$y_e$ [m]')
    ax[2].set_xlabel('time [s]')
    ax[2].grid()

    fig.suptitle(fr'Tracking-error channels vs. time -- $K_1,K_2,K_3$ = ({gains})')
    plt.tight_layout()
    plt.show()
