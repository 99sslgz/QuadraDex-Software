"""Decorative scene geometry for the simulator: tracked-target ghosts drawn on top
of the physical model. Nothing here affects physics or exported signals."""
import mujoco
import numpy as np

TARGET = np.array([0.15, 0.95, 0.55, 0.55], dtype=np.float32)
LIMIT = np.array([1.0, 0.45, 0.15, 0.65], dtype=np.float32)
HUMAN = np.array([1.0, 0.72, 0.30, 0.42], dtype=np.float32)
TIP = np.array([0.20, 0.95, 0.60, 0.80], dtype=np.float32)
AXES = [np.array([0.95, 0.2, 0.2, 0.9], np.float32), np.array([0.2, 0.9, 0.3, 0.9], np.float32),
        np.array([0.25, 0.45, 1.0, 0.9], np.float32)]


def _add(scene, geom_type, size, pos, mat=None, rgba=TARGET):
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, geom_type, np.asarray(size, float), np.asarray(pos, float),
                        (np.eye(3) if mat is None else np.asarray(mat, float)).ravel(), np.asarray(rgba, np.float32))
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1
    return geom


def _segment(scene, a, b, width, rgba):
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(np.subtract(b, a)) < 1e-6:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def draw_overlay(scene, overlay):
    """overlay: dict with optional shoulder/elbow/ghost_wrist (operator arm posture
    at robot scale), wrist + wrist_rotation (commanded tool pose), tips (list of
    world points) and limit (bool). All positions in world metres."""
    if not overlay:
        return
    colour = LIMIT if overlay.get("limit") else TARGET
    shoulder, elbow, wrist = (overlay.get(k) for k in ("shoulder", "elbow", "wrist"))
    ghost = overlay.get("ghost_wrist")
    if shoulder is not None and elbow is not None:
        _segment(scene, shoulder, elbow, .012, HUMAN)
        _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.020, 0, 0], elbow, rgba=HUMAN)
    if elbow is not None and ghost is not None:
        _segment(scene, elbow, ghost, .010, HUMAN)
        _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.016, 0, 0], ghost, rgba=HUMAN)
    if wrist is not None:
        _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.024, 0, 0], wrist, rgba=colour)
        rotation = overlay.get("wrist_rotation")
        if rotation is not None:
            rotation = np.asarray(rotation)
            for axis in range(3):
                _segment(scene, wrist, wrist+.075*rotation[:, axis], .0035, AXES[axis])
    for tip in overlay.get("tips") or []:
        _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.0085, 0, 0], tip, rgba=TIP)
