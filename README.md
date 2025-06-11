# Mobile Robot Planning

This repository contains the implementation for **Project: Mobile Robot Planning, Control and Estimation** (Subject code **H02A4a**). The goal of the project is to solve a complete mobile robot navigation pipeline, from motion planning to control and state estimation.

## Project Overview

The robot must navigate from a **start pose** to a **goal pose**, avoiding collisions with obstacles. The environment contains various capsule-shaped obstacles, and the robot is modeled as a circle for collision-checking purposes.

The robot is equipped with a **GPS sensor** that provides noisy measurements of its **position** only (not orientation). Using these noisy measurements and known control inputs, the robot must:

1. **Estimate its full state** (position and orientation) using an **Extended Kalman Filter (EKF)**.  
2. **Plan a feasible path** from start to goal using a **Probabilistic Roadmap (PRM)**.  
3. **Track the planned path** using a **trajectory tracking controller**.

## Repository Contents

| File                     | Description                                                         |
|--------------------------|---------------------------------------------------------------------|
| `mobile_robot.ipynb`     | Jupyter notebook integrating all components and demonstrating the full pipeline. |
| `prm.py`                 | Global planner using a Probabilistic Roadmap (PRM).                |
| `localPlanner.py`        | Local planner and utilities for path smoothing and interpolation.   |
| `controller.py`          | Trajectory tracking controller.                                     |
| `estimator_.py`          | EKF implementation for robot localization using GPS and control inputs. |
| `collisions.py`          | Capsule-based collision checking between robot and environment.     |
| `utils.py`               | Utility functions for transformations, geometry, and visualization. |
| `bicycle.urdf`           | URDF model for the bicycle-model robot configuration.               |
| `diffdrive.urdf`         | URDF model for the differential-drive robot configuration.          |
| `project_description.pdf`| Original project statement with specifications and objectives.      |

## Features

- **Motion Planning**: PRM-based global planner with obstacle avoidance.  
- **Trajectory Tracking**: Controller to follow planned paths robustly.  
- **State Estimation**: EKF for real-time localization from partial measurements.  
- **Collision Checking**: Efficient capsule-based geometric checks.  
- **Modular Architecture**: Clear separation of planning, control, and estimation.

## Authors

- **Èric Sánchez López** (r1026357)  
- **Jordi Beltrán Perelló** (r1032165)  
