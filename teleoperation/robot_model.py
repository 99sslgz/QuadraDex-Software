"""Build a MuJoCo scene from a configurable arm and the pinned official LEAP URDF.

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
ARM_NAMES = ["base_yaw", "shoulder_pitch", "shoulder_roll", "elbow", "wrist_yaw", "wrist_pitch", "wrist_roll"]
FINGERS = {"index": "fingertip", "middle": "fingertip_2", "ring": "fingertip_3", "thumb": "thumb_fingertip"}


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


def mesh_tip_offset(link):
    """A reproducible distal surface site, estimated from the published STL mesh."""
    visual = link.find("visual")
    filename = visual.find("geometry/mesh").get("filename")
    data = (LEAP_ASSETS / filename).read_bytes()
    triangles = int.from_bytes(data[80:84], "little")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    vertices = np.frombuffer(data, dtype=dtype, count=triangles, offset=84)["vertices"].reshape(-1, 3).astype(float)
    origin = visual.find("origin")
    rotation = rpy_matrix(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
    vertices = vertices @ rotation.T + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    distance = np.linalg.norm(vertices, axis=1)
    return np.mean(vertices[distance >= np.percentile(distance, 98)], axis=0)


def load_config(path=None):
    config = json.loads(Path(path or ROOT / "demo_arm.json").read_text(encoding="utf-8"))
    for key in ("home_rad", "lower_rad", "upper_rad"):
        value = np.asarray(config[key], dtype=float)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain seven finite joint values")
    lower, upper, home = [np.asarray(config[key]) for key in ("lower_rad", "upper_rad", "home_rad")]
    if np.any(lower >= upper) or np.any(home < lower) or np.any(home > upper):
        raise ValueError("Invalid arm joint limits or home pose")
    for key in ("upper_arm_m", "forearm_m", "base_height_m", "max_velocity_rad_s", "hand_max_velocity_rad_s", "translation_scale", "arm_fit_tolerance_m", "hand_fit_tolerance_m"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("mount_xyz_m", "mount_rpy_rad", "workspace_lower_m", "workspace_upper_m"):
        value = np.asarray(config[key], float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain three finite values")
    if np.any(np.asarray(config["workspace_lower_m"]) >= config["workspace_upper_m"]):
        raise ValueError("Invalid workspace bounds")
    return config


def scene_xml(config):
    scene = ET.Element("mujoco", model="QuadraDex demo arm + LEAP V1")
    ET.SubElement(scene, "compiler", angle="radian", autolimits="true", meshdir=str(LEAP_ASSETS))
    ET.SubElement(scene, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", offwidth="1000", offheight="800")
    ET.SubElement(visual, "headlight", diffuse="0.8 0.8 0.8", ambient="0.35 0.35 0.35")
    assets, world = ET.SubElement(scene, "asset"), ET.SubElement(scene, "worldbody")
    ET.SubElement(assets, "texture", name="grid", type="2d", builtin="checker", rgb1="0.15 0.18 0.22", rgb2="0.20 0.24 0.29", width="256", height="256")
    ET.SubElement(assets, "material", name="floor", texture="grid", texrepeat="6 6")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1")
    ET.SubElement(world, "geom", type="plane", size="2 2 .1", material="floor", contype="0", conaffinity="0")
    ET.SubElement(world, "geom", type="cylinder", pos=f"0 0 {config['base_height_m']/2}", size=f".075 {config['base_height_m']/2}", rgba=".22 .27 .35 1", contype="0", conaffinity="0")
    parent = ET.SubElement(world, "body", name="arm_base", pos=f"0 0 {config['base_height_m']}")
    origins = [[0,0,0], [0,0,0], [0,0,0], [config["upper_arm_m"],0,0], [config["forearm_m"],0,0], [0,0,0], [0,0,0]]
    axes = ["0 0 1", "0 1 0", "1 0 0", "0 1 0", "0 0 1", "0 1 0", "1 0 0"]
    for i, name in enumerate(ARM_NAMES):
        parent = ET.SubElement(parent, "body", name=name+"_body", pos=numbers(origins[i]), gravcomp="1")
        if i==3:
            ET.SubElement(parent,"site",name="elbow",pos="0 0 0",size=".005",rgba=".2 .7 1 1")
        ET.SubElement(parent, "joint", name="arm_"+name, type="hinge", axis=axes[i], range=numbers([config["lower_rad"][i],config["upper_rad"][i]]), damping="3", armature=".05")
        if i in (2,3):
            length = config["upper_arm_m"] if i == 2 else config["forearm_m"]
            ET.SubElement(parent, "geom", type="capsule", fromto=f"0 0 0 {length} 0 0", size=".033", mass="1.5", rgba=".18 .55 .74 1", contype="0", conaffinity="0")
        else:
            ET.SubElement(parent, "geom", type="sphere", size=".036", mass=".18", rgba=".25 .30 .38 1", contype="0", conaffinity="0")
    ET.SubElement(parent, "site", name="tool", pos="0 0 0", size=".006", rgba="0 1 .5 1")
    hand_root = ET.SubElement(parent, "body", name="leap_palm_lower", pos=numbers(config["mount_xyz_m"]), quat=numbers(quaternion(rpy_matrix(config["mount_rpy_rad"]))), gravcomp="1")
    urdf = ET.parse(LEAP_ASSETS/"robot.urdf").getroot()
    links = {link.get("name"):link for link in urdf.findall("link")}
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
        ET.SubElement(body,"inertial", **attrs, mass=inertial.find("mass").get("value"), fullinertia=" ".join(inertia.get(k) for k in ("ixx","iyy","izz","ixy","ixz","iyz")))
        for item in link.findall("visual"):
            filename = item.find("geometry/mesh").get("filename")
            mesh_name = Path(filename).stem
            if mesh_name not in mesh_names:
                ET.SubElement(assets,"mesh",name=mesh_name,file=filename)
                mesh_names.add(mesh_name)
            color = item.find("material/color")
            ET.SubElement(body,"geom",type="mesh",mesh=mesh_name,**transform_attributes(item.find("origin")), rgba=color.get("rgba") if color is not None else ".65 .65 .68 1", contype="0",conaffinity="0")
        if name in FINGERS.values():
            finger = next(k for k,v in FINGERS.items() if v==name)
            ET.SubElement(body,"site",name="tip_"+finger,pos=numbers(mesh_tip_offset(link)),size=".004",rgba=".2 1 .5 1")
        for joint in children.get(name,[]):
            child_name = joint.find("child").get("link")
            child = ET.SubElement(body,"body",name="leap_"+child_name,**transform_attributes(joint.find("origin")),gravcomp="1")
            limit = joint.find("limit")
            ET.SubElement(child,"joint",name="leap_"+joint.get("name"),type="hinge",axis=joint.find("axis").get("xyz"),range=f"{limit.get('lower')} {limit.get('upper')}",damping=".025",armature=".002")
            add_link(child_name,child)
    add_link("palm_lower",hand_root)
    actuators = ET.SubElement(scene,"actuator")
    for name in ARM_NAMES:
        ET.SubElement(actuators,"position",name="cmd_arm_"+name,joint="arm_"+name,kp="180",kv="24",forcerange="-100 100")
    for i in range(16):
        ET.SubElement(actuators,"position",name=f"cmd_leap_{i}",joint=f"leap_{i}",kp="12",kv=".5",forcerange="-.95 .95")
    return ET.tostring(scene,encoding="unicode")


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
        self.lower = self.model.jnt_range[ids,0]
        self.upper = self.model.jnt_range[ids,1]
        self.actuators = [self.model.actuator("cmd_"+name).id for name in self.names]
        self.home = np.r_[self.config["home_rad"], np.zeros(16)]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[self.qadr] = self.home
        self.data.ctrl[self.actuators] = self.home
        mujoco.mj_forward(self.model,self.data)
        self.ik.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model,self.ik)

    def forward(self, q):
        self.ik.qpos[self.qadr] = q
        mujoco.mj_forward(self.model,self.ik)
        return self.ik

    def measured(self):
        return self.data.qpos[self.qadr].copy()

    def advance(self, target, dt):
        target = np.asarray(target,float)
        if target.shape != (23,) or not np.isfinite(target).all():
            raise ValueError("A command must contain 23 finite arm/hand positions")
        if np.any(target < self.lower-1e-8) or np.any(target > self.upper+1e-8):
            raise ValueError("Command exceeds model joint limits")
        self.data.ctrl[self.actuators] = target
        for _ in range(max(1,min(50,round(dt/self.model.opt.timestep)))):
            mujoco.mj_step(self.model,self.data)

    def snapshot(self, path):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.26,0,.55]
        camera.distance, camera.azimuth, camera.elevation = 1.25, 125, -20
        with mujoco.Renderer(self.model,height=700,width=960) as renderer:
            renderer.update_scene(self.data,camera)
            import cv2
            cv2.imwrite(str(path),cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR))
