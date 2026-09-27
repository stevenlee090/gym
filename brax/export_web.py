"""
Export a trained Go1 policy for the in-browser joystick (web/).

Writes:
    web/go1.mjb          compiled physics model: the training model (PD gains, timestep)
                         with its visual-only geoms, meshes and skybox stripped out
    web/go1_scene.json   what the browser loads: go1.mjb plus the render meshes (float32
                         vertices, uint16 faces), both base64, and the visual geoms'
                         body, local pose and colour
    web/policy.json      observation normaliser + MLP weights (base64 float32) + env constants

--check runs the exported policy in plain CPU MuJoCo with NumPy (the same maths the
browser runs) and reports how well it tracks a few joystick commands. It also checks the
NumPy policy against brax's own inference function.

Usage:
    python export_web.py --run models/ppo_Go1JoystickFlatTerrain_seed0 --check
"""

import argparse
import base64
import json
import os
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import numpy as np
from brax.training import networks as brax_networks
from brax.training.agents.ppo import checkpoint
from mujoco_playground import registry
from mujoco_playground._src.locomotion.go1 import go1_constants as consts

import config
from evaluate import latest_checkpoint

brax_networks.KERNEL_INITIALIZER.setdefault(None, None)
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def sensor_slice(model, name):
    s = model.sensor(name)
    return [int(s.adr[0]), int(s.dim[0])]


class NumpyPolicy:
    """Deterministic brax PPO policy: normalise -> MLP (silu) -> tanh(loc)."""

    def __init__(self, norm, layers):
        self.mean, self.std = norm
        self.layers = layers

    def __call__(self, obs):
        x = (obs - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = x / (1.0 + np.exp(-x))  # silu
        return np.tanh(x[: x.shape[-1] // 2])


class CpuGo1:
    """Minimal CPU MuJoCo port of Playground's Go1 joystick step (no noise, no DR)."""

    def __init__(self, model, meta):
        self.m, self.d = model, mujoco.MjData(model)
        self.meta = meta
        self.default = np.array(meta["default_pose"])
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.meta["home_key"])
        self.d.ctrl[:] = self.default
        mujoco.mj_forward(self.m, self.d)
        self.last_act = np.zeros(12)

    def sensor(self, key):
        adr, dim = self.meta["sensors"][key]
        return self.d.sensordata[adr:adr + dim].copy()

    def obs(self, command):
        imu = self.d.site_xmat[self.meta["imu_site"]].reshape(3, 3)
        gravity = imu.T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([
            self.sensor("local_linvel"), self.sensor("gyro"), gravity,
            self.d.qpos[7:] - self.default, self.d.qvel[6:], self.last_act, command,
        ]).astype(np.float32)

    def step(self, action):
        self.d.ctrl[:] = self.default + action * self.meta["action_scale"]
        for _ in range(self.meta["n_substeps"]):
            mujoco.mj_step(self.m, self.d)
        self.last_act = action


def build_physics_model(env) -> mujoco.MjModel:
    """Training model minus everything that only matters for rendering.

    Only the floor and the four foot spheres collide, and every link declares its
    inertia explicitly, so dropping the visual geoms leaves the dynamics unchanged
    (checked below). This takes the model from ~49 MB to well under 1 MB.
    """
    from mujoco_playground._src.locomotion.go1.base import get_assets

    m = env.mj_model
    spec = mujoco.MjSpec.from_file(env.xml_path, assets=get_assets())
    for geom in list(spec.geoms):
        if geom.contype == 0 and geom.conaffinity == 0:
            spec.delete(geom)
    for geom in spec.geoms:
        geom.material = ""
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    for tex in list(spec.textures):
        spec.delete(tex)
    for mat in list(spec.materials):
        spec.delete(mat)
    m2 = spec.compile()

    # Playground edits these after compiling; copy them across.
    m2.opt.timestep = m.opt.timestep
    m2.opt.ccd_iterations = m.opt.ccd_iterations
    m2.dof_damping[:] = m.dof_damping
    m2.actuator_gainprm[:] = m.actuator_gainprm
    m2.actuator_biasprm[:] = m.actuator_biasprm

    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "dof_damping",
                  "dof_armature", "dof_frictionloss", "actuator_gainprm", "actuator_biasprm",
                  "actuator_ctrlrange", "jnt_range", "key_qpos", "sensor_adr", "site_pos"):
        if not np.allclose(getattr(m, field), getattr(m2, field)):
            raise RuntimeError(f"Stripped model differs from training model in {field}")
    for field in ("timestep", "iterations", "ls_iterations", "cone", "impratio", "solver",
                  "integrator", "noslip_iterations"):
        if getattr(m.opt, field) != getattr(m2.opt, field):
            raise RuntimeError(f"Stripped model differs in opt.{field}")
    return m2


def export_render_scene(m: mujoco.MjModel):
    """Visual geoms (groups 0-2) for three.js, posed per body at runtime."""
    geoms, mesh_ids = [], []
    for i in range(m.ngeom):
        if m.geom_group[i] > 2 or m.geom_contype[i] or m.geom_conaffinity[i]:
            continue  # hidden collision proxies, floor, feet
        mat = m.geom_matid[i]
        rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[i]
        g = {"body": int(m.geom_bodyid[i]), "type": int(m.geom_type[i]),
             "pos": [round(float(v), 6) for v in m.geom_pos[i]],
             "quat": [round(float(v), 6) for v in m.geom_quat[i]],
             "size": [round(float(v), 6) for v in m.geom_size[i]],
             "rgba": [round(float(v), 3) for v in rgba]}
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            g["mesh"] = int(m.geom_dataid[i])
            mesh_ids.append(int(m.geom_dataid[i]))
        geoms.append(g)

    chunks, meshes, offset = [], {}, 0
    for mid in sorted(set(mesh_ids)):
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        verts = m.mesh_vert[va:va + vn].astype("<f4")
        faces = m.mesh_face[fa:fa + fn].astype("<u2")
        assert vn < 65536
        pad = (-faces.nbytes) % 4
        meshes[mid] = {"v_off": offset, "v_count": int(vn * 3),
                       "f_off": offset + verts.nbytes, "f_count": int(fn * 3)}
        chunks += [verts.tobytes(), faces.tobytes(), b"\0" * pad]
        offset += verts.nbytes + faces.nbytes + pad

    scene = {"nbody": int(m.nbody), "geoms": geoms, "meshes": meshes,
             "mesh_data": base64.b64encode(b"".join(chunks)).decode()}
    return scene, len(geoms), offset


def export(args):
    ckpt = args.checkpoint or latest_checkpoint(args.run)
    ckpt = os.path.abspath(ckpt)
    params = checkpoint.load(ckpt)
    norm, pol = params[0], params[1]["params"]
    mean = np.asarray(norm.mean["state"], dtype=np.float32)
    std = np.asarray(norm.std["state"], dtype=np.float32)
    layers = [(np.asarray(pol[f"hidden_{i}"]["kernel"], dtype=np.float32),
               np.asarray(pol[f"hidden_{i}"]["bias"], dtype=np.float32))
              for i in range(len(pol))]

    env = registry.load(config.ENV_NAME)
    cfg = env._config
    m = env.mj_model
    meta = {
        "env": config.ENV_NAME,
        "checkpoint": os.path.basename(ckpt),
        "env_steps": int(os.path.basename(ckpt)),
        "ctrl_dt": float(cfg.ctrl_dt),
        "sim_dt": float(cfg.sim_dt),
        "n_substeps": int(round(cfg.ctrl_dt / cfg.sim_dt)),
        "action_scale": float(cfg.action_scale),
        "home_key": int(m.keyframe("home").id),
        "default_pose": [float(v) for v in m.keyframe("home").qpos[7:]],
        "imu_site": int(m.site("imu").id),
        "torso_body": int(m.site_bodyid[m.site("imu").id]),
        "sensors": {
            "local_linvel": sensor_slice(m, consts.LOCAL_LINVEL_SENSOR),
            "gyro": sensor_slice(m, consts.GYRO_SENSOR),
        },
        "command_max": [float(v) for v in cfg.command_config.a],
    }

    os.makedirs(WEB_DIR, exist_ok=True)
    mjb = os.path.join(WEB_DIR, "go1.mjb")
    physics = build_physics_model(env)
    mujoco.mj_saveModel(physics, mjb, None)
    # Artifact hosting serves JSON but not raw binaries, so the compiled model and the
    # render meshes travel base64-encoded inside go1_scene.json.
    scene, ngeom, mesh_bytes = export_render_scene(m)
    with open(mjb, "rb") as f:
        scene["mjb"] = base64.b64encode(f.read()).decode()
    with open(os.path.join(WEB_DIR, "go1_scene.json"), "w") as f:
        json.dump(scene, f)
    print(f"Render scene: {ngeom} visual geoms, {mesh_bytes/1e6:.1f} MB of meshes")
    # A reference observation/action pair so the browser can check its policy port.
    ref_sim = CpuGo1(physics, meta)
    for _ in range(25):
        ref_sim.step(NumpyPolicy((mean, std), layers)(ref_sim.obs(np.array([0.5, 0.2, 0.3], np.float32))))
    ref_obs = ref_sim.obs(np.array([0.5, 0.2, 0.3], np.float32))
    ref_act = NumpyPolicy((mean, std), layers)(ref_obs)
    policy_json = {
        **meta,
        "ref_obs": [float(v) for v in ref_obs],
        "ref_act": [float(v) for v in ref_act],
        "obs_mean": b64(mean), "obs_std": b64(std),
        "layers": [{"in": int(w.shape[0]), "out": int(w.shape[1]), "w": b64(w), "b": b64(b)}
                   for w, b in layers],
    }
    with open(os.path.join(WEB_DIR, "policy.json"), "w") as f:
        json.dump(policy_json, f)
    print(f"Exported {os.path.basename(ckpt)} -> web/go1.mjb "
          f"({os.path.getsize(mjb)/1e6:.1f} MB), web/policy.json "
          f"({os.path.getsize(os.path.join(WEB_DIR, 'policy.json'))/1e6:.2f} MB)")

    vendor_mujoco()

    if args.check:
        check(mjb, meta, NumpyPolicy((mean, std), layers), ckpt)


def vendor_mujoco():
    """Local copy of the MuJoCo WebAssembly build, matching the Python version.

    Published artifact pages cannot fetch binaries from a CDN, so the page loads these
    from web/vendor/ instead.
    """
    import urllib.request

    vendor = os.path.join(WEB_DIR, "vendor")
    os.makedirs(vendor, exist_ok=True)
    base = f"https://cdn.jsdelivr.net/npm/@mujoco/mujoco@{mujoco.__version__}/"
    for name in ("mujoco.js", "mujoco.wasm"):
        path = os.path.join(vendor, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(base + name, path)
    print(f"MuJoCo {mujoco.__version__} WebAssembly build in web/vendor/")


def check(mjb, meta, policy, ckpt):
    import jax
    brax_policy = checkpoint.load_policy(ckpt, deterministic=True)
    model = mujoco.MjModel.from_binary_path(mjb)
    sim = CpuGo1(model, meta)

    # 1. NumPy policy matches brax on real observations.
    sim.reset()
    worst = 0.0
    for t in range(50):
        o = sim.obs(np.array([0.8, 0.0, 0.3], np.float32))
        a_np = policy(o)
        a_bx, _ = brax_policy({"state": o, "privileged_state": np.zeros(123, np.float32)},
                              jax.random.PRNGKey(0))
        worst = max(worst, float(np.max(np.abs(a_np - np.asarray(a_bx)))))
        sim.step(a_np)
    print(f"  NumPy vs brax policy: max |Δaction| = {worst:.2e}")

    # 2. Sim-to-sim: does it track commands in CPU MuJoCo?
    scripts = {
        "forward 1.0 m/s": [1.0, 0.0, 0.0],
        "backward 0.8 m/s": [-0.8, 0.0, 0.0],
        "strafe left 0.6 m/s": [0.0, 0.6, 0.0],
        "turn left 1.0 rad/s": [0.0, 0.0, 1.0],
        "arc (0.8, 0, -0.8)": [0.8, 0.0, -0.8],
    }
    print("  CPU MuJoCo tracking (mean over 2-5 s, from reset):")
    for name, cmd in scripts.items():
        sim.reset()
        cmd = np.array(cmd, np.float32)
        vs = []
        for t in range(250):
            sim.step(policy(sim.obs(cmd)))
            if t >= 100:
                vs.append(np.concatenate([sim.sensor("local_linvel")[:2], sim.sensor("gyro")[2:]]))
        v = np.mean(vs, axis=0)
        up = sim.d.qpos[2] > 0.15
        print(f"    {name:22s} asked {cmd}  got [{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]"
              f"  {'upright' if up else 'FELL'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Go1 policy for the browser")
    parser.add_argument("--run", default=os.path.join(
        config.MODEL_DIR, f"ppo_{config.ENV_NAME}_seed{config.SEED}"))
    parser.add_argument("--checkpoint", default=None, help="Checkpoint dir (default: latest)")
    parser.add_argument("--check", action="store_true",
                        help="Verify the export in CPU MuJoCo before shipping it")
    export(parser.parse_args())
