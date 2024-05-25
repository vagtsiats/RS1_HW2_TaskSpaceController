import pinocchio as pin
import numpy as np


def RotX(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.eye(3, 3)
    R[1, 1] = ct
    R[1, 2] = -st
    R[2, 1] = st
    R[2, 2] = ct
    return R


def RotY(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.eye(3, 3)
    R[0, 0] = ct
    R[0, 2] = st
    R[2, 0] = -st
    R[2, 2] = ct
    return R


def RotZ(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    R = np.eye(3, 3)
    R[0, 0] = ct
    R[0, 1] = -st
    R[1, 0] = st
    R[1, 1] = ct
    return R


# function to compute FK for all frames/joints
def fk_all(model, data, q, v=None):
    if v is not None:
        pin.forwardKinematics(model, data, q, v)  # FK and Forward Velocities
    else:
        pin.forwardKinematics(model, data, q)  # FK
    pin.updateFramePlacements(model, data)  # Update frames


def interpolate_translation(start, goal, s):
    return start + s * (goal - start)


def damped_pseudoinverse(jac, l=0.001):
    m, n = jac.shape
    if n >= m:
        return jac.T @ np.linalg.inv(jac @ jac.T + l * l * np.eye(m))
    return np.linalg.inv(jac.T @ jac + l * l * np.eye(n)) @ jac.T
