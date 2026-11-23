"""Build the MuJoCo follower scene: a configurable 7-DOF demo arm carrying the
pinned official LEAP Hand V1 model on a bench, with the project's evaluation
objects for scale.

The LEAP joint frames, mesh placements, inertias and limits come from the URDF.
No correspondence between URDF traversal order and motor order is assumed.
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
LEAP_ASSETS = ROOT / "assets" / "leap_v1"
# Human-like shoulder (3) / elbow (1) / forearm-wrist (3) arrangement.
ARM_NAMES = ["shoulder_yaw", "shoulder_pitch", "upper_arm_roll", "elbow",
             "forearm_roll", "wrist_pitch", "wrist_yaw"]
FINGERS = {"index": "fingertip", "middle": "fingertip_2", "ring": "fingertip_3", "thumb": "thumb_fingertip"}

# Collision groups: robot links never collide with each other, only with the
# bench and the movable objects.
ROBOT_COLLISION = {"contype": "2", "conaffinity": "1"}
WORLD_COLLISION = {"contype": "1", "conaffinity": "3"}
NO_COLLISION = {"contype": "0", "conaffinity": "0"}

DEFAULTS = {
    "bench_height_m": 0.76,
    "bench_size_m": [1.20, 0.80],
    "bench_center_xy_m": [0.42, 0.0],
    "scene_objects": True,
    "hand_mount_offset_m": 0.05,
    "hand_kp": 8.0,
    "arm_kp": 400.0,
    "match_start_pose": True,
    "align_speed_scale": 0.3,
}

# Colours of the LEAP Hand links: dark servo assemblies, light printed parts.
LEAP_MATERIAL = {"palm_lower": "leap_print", "mcp_joint": "leap_motor", "pip": "leap_bracket",
                 "dip": "leap_motor", "fingertip": "leap_tip", "thumb_pip": "leap_bracket",
                 "thumb_dip": "leap_motor", "thumb_fingertip": "leap_tip"}


def numbers(value):
    return " ".join(f"{float(x):.12g}" for x in np.asarray(value).flat)


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def quaternion(rotation):
    out = np.empty(4)
    mujoco.mju_mat2Quat(out, np.asarray(rotation, dtype=float).ravel())
    return out


def transform_attributes(element):
    if element is None:
        return {"pos": "0 0 0", "quat": "1 0 0 0"}
    xyz = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    return {"pos": numbers(xyz), "quat": numbers(quaternion(rpy_matrix(rpy)))}


def stl_vertices(filename):
    data = (LEAP_ASSETS / filename).read_bytes()
    triangles = int.from_bytes(data[80:84], "little")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(data, dtype=dtype, count=triangles, offset=84)["vertices"].reshape(-1, 3).astype(float)


def mesh_tip_offset(link):
    """A reproducible distal surface site, estimated from the published STL mesh."""
    visual = link.find("visual")
    vertices = stl_vertices(visual.find("geometry/mesh").get("filename"))
    origin = visual.find("origin")
    rotation = rpy_matrix(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
    vertices = vertices @ rotation.T + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    distance = np.linalg.norm(vertices, axis=1)
    return np.mean(vertices[distance >= np.percentile(distance, 98)], axis=0)


def palm_extent(link):
    """Bounding box of the palm mesh in the palm_lower link frame."""
    visual = link.find("visual")
    vertices = stl_vertices(visual.find("geometry/mesh").get("filename"))
    origin = visual.find("origin")
    rotation = rpy_matrix(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
    vertices = vertices @ rotation.T + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    return vertices.min(axis=0), vertices.max(axis=0)


def load_config(path=None):
    config = {**DEFAULTS, **json.loads(Path(path or ROOT / "demo_arm.json").read_text(encoding="utf-8"))}
    for key in ("home_rad", "lower_rad", "upper_rad"):
        value = np.asarray(config[key], dtype=float)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain seven finite joint values")
    lower, upper, home = [np.asarray(config[key]) for key in ("lower_rad", "upper_rad", "home_rad")]
    if np.any(lower >= upper) or np.any(home < lower) or np.any(home > upper):
        raise ValueError("Invalid arm joint limits or home pose")
    for key in ("upper_arm_m", "forearm_m", "base_height_m", "max_velocity_rad_s", "hand_max_velocity_rad_s",
                "translation_scale", "arm_fit_tolerance_m", "hand_fit_tolerance_m", "bench_height_m",
                "hand_kp", "arm_kp"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if config["bench_height_m"] >= config["base_height_m"]:
        raise ValueError("The arm base must be above the bench")
    for key in ("workspace_lower_m", "workspace_upper_m"):
        value = np.asarray(config[key], float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain three finite values")
    if np.any(np.asarray(config["workspace_lower_m"]) >= config["workspace_upper_m"]):
        raise ValueError("Invalid workspace bounds")
    if not 0 < float(config["align_speed_scale"]) <= 1:
        raise ValueError("align_speed_scale must be in (0, 1]")
    if config.get("limit_policy", "constrain") not in ("constrain", "reject"):
        raise ValueError("limit_policy must be 'constrain' or 'reject'")
    return config


def _assets(scene):
    assets = ET.SubElement(scene, "asset")
    ET.SubElement(assets, "texture", type="skybox", builtin="gradient", rgb1="0.42 0.50 0.60",
                  rgb2="0.07 0.08 0.10", width="512", height="3072")
    ET.SubElement(assets, "texture", name="floor", type="2d", builtin="checker", rgb1="0.20 0.22 0.25",
                  rgb2="0.16 0.18 0.21", width="512", height="512", mark="edge", markrgb="0.28 0.30 0.34")
    ET.SubElement(assets, "material", name="floor", texture="floor", texrepeat="10 10", reflectance="0.12")
    ET.SubElement(assets, "texture", name="bench", type="2d", builtin="flat", rgb1="0.55 0.57 0.59",
                  rgb2="0.52 0.54 0.56", width="256", height="256", mark="random", random="0.015",
                  markrgb="0.58 0.60 0.62")
    materials = {
        "bench": dict(texture="bench", texrepeat="4 4", specular="0.25", shininess="0.3", reflectance="0.04"),
        "bench_frame": dict(rgba="0.18 0.19 0.21 1", specular="0.5", shininess="0.6"),
        "tray": dict(rgba="0.13 0.20 0.28 1", specular="0.1", shininess="0.1"),
        "pedestal": dict(rgba="0.24 0.26 0.29 1", specular="0.6", shininess="0.7"),
        "arm_shell": dict(rgba="0.90 0.91 0.92 1", specular="0.55", shininess="0.75"),
        "arm_joint": dict(rgba="0.17 0.18 0.20 1", specular="0.6", shininess="0.8"),
        "arm_accent": dict(rgba="0.10 0.52 0.86 1", specular="0.6", shininess="0.8", emission="0.15"),
        "flange": dict(rgba="0.62 0.64 0.67 1", specular="0.9", shininess="0.9"),
        "leap_motor": dict(rgba="0.12 0.12 0.13 1", specular="0.45", shininess="0.55"),
        "leap_bracket": dict(rgba="0.22 0.23 0.25 1", specular="0.35", shininess="0.4"),
        "leap_print": dict(rgba="0.86 0.86 0.84 1", specular="0.2", shininess="0.25"),
        "leap_tip": dict(rgba="0.93 0.93 0.91 1", specular="0.15", shininess="0.2"),
        "bottle": dict(rgba="0.55 0.78 0.95 0.55", specular="0.9", shininess="0.95", reflectance="0.05"),
        "bottle_cap": dict(rgba="0.10 0.35 0.75 1", specular="0.4"),
        "water": dict(rgba="0.35 0.62 0.90 0.35"),
        "cube_core": dict(rgba="0.05 0.05 0.06 1"),
        "airpods": dict(rgba="0.97 0.97 0.96 1", specular="0.7", shininess="0.9"),
        "phone": dict(rgba="0.30 0.29 0.27 1", specular="0.8", shininess="0.9"),
        "phone_glass": dict(rgba="0.03 0.03 0.04 1", specular="1", shininess="1", reflectance="0.1"),
    }
    for name, attrs in materials.items():
        ET.SubElement(assets, "material", name=name, **attrs)
    for colour, rgba in {"white": ".95 .95 .95 1", "yellow": ".98 .84 .10 1", "red": ".80 .08 .10 1",
                         "orange": "1 .45 .05 1", "blue": ".05 .30 .80 1", "green": ".05 .62 .25 1"}.items():
        ET.SubElement(assets, "material", name="cube_"+colour, rgba=rgba, specular="0.5", shininess="0.6")
    return assets


def _bench(world, config):
    height = config["bench_height_m"]
    length, width = config["bench_size_m"]
    cx, cy = config["bench_center_xy_m"]
    top = 0.025
    bench = ET.SubElement(world, "body", name="bench", pos=f"{cx} {cy} 0")
    ET.SubElement(bench, "geom", name="bench_top", type="box", size=f"{length/2} {width/2} {top}",
                  pos=f"0 0 {height-top}", material="bench", friction="1 0.01 0.001", **WORLD_COLLISION)
    for sx in (-1, 1):
        for sy in (-1, 1):
            ET.SubElement(bench, "geom", type="box", size=f".025 .025 {(height-2*top)/2}",
                          pos=f"{sx*(length/2-.05)} {sy*(width/2-.05)} {(height-2*top)/2}",
                          material="bench_frame", **NO_COLLISION)
    ET.SubElement(bench, "geom", type="box", size=f"{length/2-.05} {width/2-.05} .012",
                  pos=f"0 0 {0.18}", material="bench_frame", **NO_COLLISION)
    # Padded catch tray (concept design) in front of the arm.
    ET.SubElement(bench, "geom", name="catch_tray", type="box", size=".22 .30 .004",
                  pos=f"{0.60-cx} {-cy} {height+0.004}", material="tray", **NO_COLLISION)


def _objects(world, config):
    """Defined manipulation object set (spec 4.5), approximate dimensions."""
    z = config["bench_height_m"]
    contact = dict(friction="1.2 0.02 0.002", condim="4", **WORLD_COLLISION)
    bottle = ET.SubElement(world, "body", name="water_bottle", pos=f"0.50 -0.27 {z+0.1051}")
    ET.SubElement(bottle, "freejoint")
    ET.SubElement(bottle, "geom", type="cylinder", size=".0330 .095", material="bottle", mass=".52", **contact)
    ET.SubElement(bottle, "geom", type="cylinder", size=".0325 .070", pos="0 0 -.02", material="water", **NO_COLLISION)
    ET.SubElement(bottle, "geom", type="cylinder", size=".016 .008", pos="0 0 .102", material="bottle_cap", mass=".005", **contact)
    cube = ET.SubElement(world, "body", name="rubiks_cube", pos=f"0.62 0.08 {z+0.0286}")
    ET.SubElement(cube, "freejoint")
    s = 0.0285
    ET.SubElement(cube, "geom", type="box", size=f"{s} {s} {s}", material="cube_core", mass=".09", **contact)
    for colour, axis, sign in [("white", 2, 1), ("yellow", 2, -1), ("red", 0, 1), ("orange", 0, -1),
                               ("blue", 1, 1), ("green", 1, -1)]:
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                pos, size = np.zeros(3), np.full(3, .0082)
                pos[axis], size[axis] = sign*(s+.0004), .0006
                others = [k for k in range(3) if k != axis]
                pos[others[0]], pos[others[1]] = i*.019, j*.019
                ET.SubElement(cube, "geom", type="box", size=numbers(size), pos=numbers(pos),
                              material="cube_"+colour, **NO_COLLISION)
    case = ET.SubElement(world, "body", name="airpods_case", pos=f"0.50 0.27 {z+0.0109}")
    ET.SubElement(case, "freejoint")
    ET.SubElement(case, "geom", type="ellipsoid", size=".0305 .0108 .0233", euler="1.5708 0 0", material="airpods",
                  mass=".045", **contact)
    phone = ET.SubElement(world, "body", name="phone", pos=f"0.70 -0.10 {z+0.0042}", euler="0 0 0.35")
    ET.SubElement(phone, "freejoint")
    ET.SubElement(phone, "geom", type="box", size=".0357 .0745 .0041", material="phone", mass=".199", **contact)
    ET.SubElement(phone, "geom", type="box", size=".0345 .0733 .0002", pos="0 0 .0041", material="phone_glass", **NO_COLLISION)


def _arm(world, config):
    """Seven-joint arm with visible actuator housings aligned with each axis."""
    L1, L2 = config["upper_arm_m"], config["forearm_m"]
    base_z, bench_z = config["base_height_m"], config["bench_height_m"]
    # Pedestal from the bench to the shoulder.
    ET.SubElement(world, "geom", type="cylinder", pos=f"0 0 {bench_z+.008}", size=".11 .008", material="flange", **NO_COLLISION)
    column = base_z-bench_z-.07
    ET.SubElement(world, "geom", type="cylinder", pos=f"0 0 {bench_z+column/2}", size=f".060 {column/2}",
                  material="pedestal", **NO_COLLISION)
    ET.SubElement(world, "geom", type="cylinder", pos=f"0 0 {bench_z+column-.01}", size=".064 .006",
                  material="arm_accent", **NO_COLLISION)
    parent = ET.SubElement(world, "body", name="arm_base", pos=f"0 0 {base_z}")
    specs = [
        # name, joint axis, body offset, housing (fromto) and shell
        ("shoulder_yaw", "0 0 1", [0, 0, 0]),
        ("shoulder_pitch", "0 -1 0", [0, 0, 0]),
        ("upper_arm_roll", "1 0 0", [0, 0, 0]),
        ("elbow", "0 -1 0", [L1, 0, 0]),
        ("forearm_roll", "1 0 0", [0, 0, 0]),
        ("wrist_pitch", "0 -1 0", [L2, 0, 0]),
        ("wrist_yaw", "0 0 1", [0, 0, 0]),
    ]
    for i, (name, axis, offset) in enumerate(specs):
        parent = ET.SubElement(parent, "body", name=name+"_body", pos=numbers(offset), gravcomp="1")
        ET.SubElement(parent, "joint", name="arm_"+name, type="hinge", axis=axis,
                      range=numbers([config["lower_rad"][i], config["upper_rad"][i]]), damping="2", armature=".05")
        g = dict(**ROBOT_COLLISION)
        if name == "shoulder_yaw":
            ET.SubElement(parent, "geom", type="cylinder", fromto="0 0 -.075 0 0 -.02", size=".066", material="arm_joint", mass="1.0", **NO_COLLISION)
        elif name == "shoulder_pitch":
            ET.SubElement(parent, "geom", type="cylinder", fromto="0 -.072 0 0 .072 0", size=".062", material="arm_joint", mass="1.2", **g)
            ET.SubElement(parent, "geom", type="cylinder", fromto="0 .072 0 0 .078 0", size=".052", material="arm_accent", mass=".01", **NO_COLLISION)
            ET.SubElement(parent, "geom", type="cylinder", fromto="0 -.078 0 0 -.072 0", size=".052", material="arm_accent", mass=".01", **NO_COLLISION)
        elif name == "upper_arm_roll":
            ET.SubElement(parent, "geom", type="capsule", fromto=f".07 0 0 {L1-.07} 0 0", size=".046", material="arm_shell", mass="1.4", **g)
            ET.SubElement(parent, "geom", type="cylinder", fromto=".055 0 0 .085 0 0", size=".050", material="arm_joint", mass=".1", **NO_COLLISION)
        elif name == "elbow":
            ET.SubElement(parent, "site", name="elbow", pos="0 0 0", size=".006", rgba=".2 .7 1 0")
            ET.SubElement(parent, "geom", type="cylinder", fromto="0 -.058 0 0 .058 0", size=".052", material="arm_joint", mass=".9", **g)
            ET.SubElement(parent, "geom", type="cylinder", fromto="0 .058 0 0 .064 0", size=".043", material="arm_accent", mass=".01", **NO_COLLISION)
        elif name == "forearm_roll":
            ET.SubElement(parent, "geom", type="capsule", fromto=f".065 0 0 {L2-.055} 0 0", size=".038", material="arm_shell", mass="1.0", **g)
            ET.SubElement(parent, "geom", type="cylinder", fromto=".05 0 0 .078 0 0", size=".042", material="arm_joint", mass=".1", **NO_COLLISION)
        elif name == "wrist_pitch":
            ET.SubElement(parent, "geom", type="cylinder", fromto="0 -.046 0 0 .046 0", size=".040", material="arm_joint", mass=".5", **g)
        elif name == "wrist_yaw":
            ET.SubElement(parent, "site", name="tool", pos="0 0 0", size=".006", rgba="0 1 .5 0")
            ET.SubElement(parent, "geom", type="cylinder", fromto="0 0 -.036 0 0 .036", size=".033", material="arm_joint", mass=".3", **g)
            mount = config["hand_mount_offset_m"]
            ET.SubElement(parent, "geom", type="cylinder", fromto=f".02 0 0 {mount-.004} 0 0", size=".030", material="flange", mass=".08", **g)
            ET.SubElement(parent, "geom", type="cylinder", fromto=f"{mount-.006} 0 0 {mount} 0 0", size=".040", material="arm_accent", mass=".02", **NO_COLLISION)
    return parent


def scene_xml(config):
    scene = ET.Element("mujoco", model="QuadraDex demo arm + LEAP V1")
    ET.SubElement(scene, "compiler", angle="radian", autolimits="true", meshdir=str(LEAP_ASSETS))
    ET.SubElement(scene, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast",
                  cone="elliptic", impratio="10")
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080", azimuth="-150", elevation="-22", fovy="40")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="8")
    ET.SubElement(visual, "headlight", diffuse="0.35 0.35 0.35", ambient="0.22 0.22 0.24", specular="0.05 0.05 0.05")
    ET.SubElement(visual, "map", znear="0.01", zfar="40", shadowclip="1.1", haze="0.25")
    ET.SubElement(visual, "rgba", haze="0.10 0.11 0.13 1")
    assets = _assets(scene)
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(world, "light", name="key", pos="1.2 -1.0 2.6", dir="-0.45 0.35 -1", directional="true",
                  castshadow="true", diffuse="0.62 0.60 0.58", specular="0.25 0.25 0.25")
    ET.SubElement(world, "light", name="fill", pos="-1 1.2 2.2", dir="0.4 -0.5 -1", directional="true",
                  castshadow="false", diffuse="0.25 0.27 0.32", specular="0 0 0")
    ET.SubElement(world, "light", name="rim", pos="0.4 0 2.4", dir="0 0 -1", castshadow="false",
                  diffuse="0.20 0.20 0.20", specular="0.1 0.1 0.1", cutoff="60", attenuation="1 0 0")
    ET.SubElement(world, "geom", name="floor", type="plane", size="4 4 .1", material="floor", **WORLD_COLLISION)
    _bench(world, config)
    if config["scene_objects"]:
        _objects(world, config)
    base_z = config["base_height_m"]
    # Fixed viewpoints, cycled with [ and ] in the viewer.
    ET.SubElement(world, "camera", name="operator_view", mode="targetbody", target="leap_palm_lower",
                  pos=f"-0.75 0.30 {base_z+0.45}")
    ET.SubElement(world, "camera", name="hand_close", mode="targetbody", target="leap_palm_lower",
                  pos=f"0.95 -0.45 {base_z+0.10}", fovy="28")
    ET.SubElement(world, "camera", name="side", mode="targetbody", target="arm_base",
                  pos=f"0.40 -1.55 {base_z+0.05}", fovy="42")
    ET.SubElement(world, "camera", name="top", pos=f"0.42 0 {base_z+1.6}", xyaxes="0 -1 0 1 0 0", fovy="45")

    wrist = _arm(world, config)
    urdf = ET.parse(LEAP_ASSETS/"robot.urdf").getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    # Palm-down right hand: LEAP palm frame x = finger direction, y = toward the
    # thumb, -z = palmar side. Align it with the tool frame and put the wrist
    # end of the palm on the flange.
    low, high = palm_extent(links["palm_lower"])
    mount = np.array([config["hand_mount_offset_m"]-low[0], -(low[1]+high[1])/2, -(low[2]+high[2])/2])
    hand_root = ET.SubElement(wrist, "body", name="leap_palm_lower", pos=numbers(mount), gravcomp="1")
    children = {}
    for joint in urdf.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    mesh_names = set()

    def add_link(name, body):
        link = links[name]
        inertial = link.find("inertial")
        inertia = inertial.find("inertia")
        attrs = transform_attributes(inertial.find("origin"))
        # URDF inertias are expressed at the provided inertial origin (zero RPY here).
        attrs.pop("quat")
        ET.SubElement(body, "inertial", **attrs, mass=inertial.find("mass").get("value"),
                      fullinertia=" ".join(inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        for item in link.findall("visual"):
            filename = item.find("geometry/mesh").get("filename")
            mesh_name = Path(filename).stem
            if mesh_name not in mesh_names:
                ET.SubElement(assets, "mesh", name=mesh_name, file=filename, smoothnormal="true", maxhullvert="64")
                mesh_names.add(mesh_name)
            ET.SubElement(body, "geom", name="geom_"+name, type="mesh", mesh=mesh_name, **transform_attributes(item.find("origin")),
                          material=LEAP_MATERIAL.get(name.rsplit("_", 1)[0] if name[-1].isdigit() else name, "leap_print"),
                          friction="1.2 0.02 0.002", condim="4", **ROBOT_COLLISION)
        if name in FINGERS.values():
            finger = next(k for k, v in FINGERS.items() if v == name)
            ET.SubElement(body, "site", name="tip_"+finger, pos=numbers(mesh_tip_offset(link)), size=".004", rgba=".2 1 .5 0")
        for joint in children.get(name, []):
            child_name = joint.find("child").get("link")
            child = ET.SubElement(body, "body", name="leap_"+child_name, **transform_attributes(joint.find("origin")), gravcomp="1")
            limit = joint.find("limit")
            ET.SubElement(child, "joint", name="leap_"+joint.get("name"), type="hinge", axis=joint.find("axis").get("xyz"),
                          range=f"{limit.get('lower')} {limit.get('upper')}", damping=".03", armature=".002", frictionloss=".002")
            add_link(child_name, child)
    add_link("palm_lower", hand_root)
    actuators = ET.SubElement(scene, "actuator")
    for name in ARM_NAMES:
        kp = config["arm_kp"]
        ET.SubElement(actuators, "position", name="cmd_arm_"+name, joint="arm_"+name, kp=f"{kp}",
                      kv=f"{2*np.sqrt(kp)*1.1:.3f}", forcerange="-150 150")
    for i in range(16):
        kp = config["hand_kp"]
        ET.SubElement(actuators, "position", name=f"cmd_leap_{i}", joint=f"leap_{i}", kp=f"{kp}",
                      kv=f"{0.12*np.sqrt(kp):.4f}", forcerange="-.95 .95")
    return ET.tostring(scene, encoding="unicode")


class RobotModel:
    def __init__(self, config_path=None):
        self.config = load_config(config_path)
        self.model = mujoco.MjModel.from_xml_string(scene_xml(self.config))
        self.data = mujoco.MjData(self.model)
        self.ik = mujoco.MjData(self.model)
        self.arm_joint_names = ["arm_"+name for name in ARM_NAMES]
        self.hand_joint_names = [f"leap_{i}" for i in range(16)]
        self.names = self.arm_joint_names+self.hand_joint_names
        ids = [self.model.joint(name).id for name in self.names]
        self.qadr = self.model.jnt_qposadr[ids]
        self.dofs = self.model.jnt_dofadr[ids]
        self.lower = self.model.jnt_range[ids, 0]
        self.upper = self.model.jnt_range[ids, 1]
        self.actuators = [self.model.actuator("cmd_"+name).id for name in self.names]
        self.home = np.r_[self.config["home_rad"], np.zeros(16)]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.home
        self.data.ctrl[self.actuators] = self.home
        mujoco.mj_forward(self.model, self.data)
        self.ik.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.ik)

    def reset_objects(self):
        """Return the movable objects to their initial poses, keeping the robot state."""
        robot_q, robot_v = self.data.qpos[self.qadr].copy(), self.data.qvel[self.dofs].copy()
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0
        self.data.qpos[self.qadr], self.data.qvel[self.dofs] = robot_q, robot_v
        mujoco.mj_forward(self.model, self.data)

    def forward(self, q):
        self.ik.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.ik)
        mujoco.mj_comPos(self.model, self.ik)
        return self.ik

    def measured(self):
        return self.data.qpos[self.qadr].copy()

    def advance(self, target, dt):
        """Step the simulation toward `target`. Never raises for a limit: the
        servo setpoint is clamped into the joint range (a held pose read back
        from the simulation can sit fractionally past a limit after contact or
        servo overshoot). A malformed target holds the current setpoint."""
        target = np.asarray(target, float)
        if target.shape == (23,) and np.isfinite(target).all():
            self.data.ctrl[self.actuators] = np.clip(target, self.lower, self.upper)
        for _ in range(max(1, min(50, round(dt/self.model.opt.timestep)))):
            mujoco.mj_step(self.model, self.data)

    def snapshot(self, path, overlay=None, camera_name=None, size=(1280, 800)):
        import cv2
        from teleoperation.visuals import draw_overlay
        camera = mujoco.MjvCamera()
        if camera_name:
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = self.model.camera(camera_name).id
        else:
            default_camera(camera, self.config)
        option = mujoco.MjvOption()
        with mujoco.Renderer(self.model, height=size[1], width=size[0]) as renderer:
            renderer.update_scene(self.data, camera, option)
            if overlay:
                draw_overlay(renderer.scene, overlay)
            cv2.imwrite(str(path), cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))


def default_camera(camera, config, close=False):
    """Three-quarter view from behind the operator's side of the robot."""
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if close:
        camera.distance, camera.azimuth, camera.elevation = .55, -140, -28
    else:
        camera.lookat[:] = [.36, 0, config["base_height_m"]-.22]
        camera.distance, camera.azimuth, camera.elevation = 1.55, -145, -24
