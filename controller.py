import numpy as np
import matplotlib.pyplot as plt

def wrap_angle(a):
    """Map any angle to (-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

def pose_error(actual, desired):
    """Return (phi_e, x_e, y_e) as in Eq. (13.30)."""
    phi,  x,  y  = actual
    phi_d, x_d, y_d = desired

    dphi = wrap_angle(phi - phi_d)
    dx, dy = x - x_d, y - y_d

    c, s = np.cos(phi_d), np.sin(phi_d)
    x_e =  c*dx + s*dy
    y_e = -s*dx + c*dy
    return dphi, x_e, y_e

def tracking_controller(actual_pose, desired_pose, desired_ctrl,
                        gains=(2.0, 6.0, 1.0), eps=1e-6):
    """
    Non-linear trajectory-tracking controller of Modern Robotics Eq. (13.31).

    Parameters
    ----------
    actual_pose  : (phi, x, y)
    desired_pose : (phi_d, x_d, y_d)
    desired_ctrl : (v_d,  omega_d)
    gains        : (k1, k2, k3)
    returns      : (v, omega)
    """
    k1, k2, k3 = gains
    phi_e, x_e, y_e = pose_error(actual_pose, desired_pose)

    # guard against division by zero when cos(phi_e) ~ 0
    if abs(np.cos(phi_e)) < eps:
        phi_e = np.sign(phi_e)*(np.pi/2 - eps)

    v_d, w_d = desired_ctrl
    v  = (v_d - k1*abs(v_d)*(x_e + y_e*np.tan(phi_e))) / np.cos(phi_e)
    w  = w_d - (k2*v_d*y_e + k3*abs(v_d)*np.tan(phi_e))*np.cos(phi_e)**2
    return v, w


if __name__ == "__main__": 
    # --- make a reference trajectory ------------------------------------------
    T         = 8.0                # seconds
    dt        = 0.02
    t_grid    = np.arange(0, T+dt, dt)
    v_d       = 0.30*np.ones_like(t_grid)   # 0.30 m/s
    w_d       = np.zeros_like(t_grid)       # straight
    x_d       = v_d.cumsum()*dt
    y_d       = np.zeros_like(t_grid)
    phi_d     = np.zeros_like(t_grid)

    # --- simulate real robot ---------------------------------------------------
    x  = np.zeros_like(x_d);  y  = np.zeros_like(y_d);  phi = np.zeros_like(phi_d)
    for k in range(len(t_grid)-1):
        desired_pose  = (phi_d[k], x_d[k], y_d[k])
        desired_ctrl  = (v_d[k],  w_d[k])

        v, w = tracking_controller((phi[k], x[k], y[k]), desired_pose, desired_ctrl)

        # Euler step –  differential-drive kinematics
        phi[k+1] = phi[k] + w*dt
        x[k+1]   = x[k]   + v*np.cos(phi[k])*dt
        y[k+1]   = y[k]   + v*np.sin(phi[k])*dt

    # --- plot ------------------------------------------------------------------
    plt.plot(x_d, y_d, '--', label='reference')
    plt.plot(x,   y,   label='actual')
    plt.axis('equal');  plt.legend();  plt.xlabel('x / m');  plt.ylabel('y / m')
    plt.title('Trajectory–tracking test');  plt.show()
