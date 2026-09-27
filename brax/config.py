"""
Configuration for GPU-accelerated PPO on MuJoCo Playground locomotion tasks.

Hyperparameters come from Playground's tuned defaults (locomotion_params.brax_ppo_config);
anything set in OVERRIDES replaces them.
"""

ENV_NAME = "Go1JoystickFlatTerrain"
SEED = 0

# Domain randomization (friction, mass, joint armature/damping) — improves robustness.
DOMAIN_RANDOMIZATION = True

# Overrides applied on top of the tuned Playground config. The tuned default for Go1
# is 200M steps; ~100M already produces a solid trot on a single 3080.
OVERRIDES = {
    "num_timesteps": 100_000_000,
    "num_evals": 20,
}

# Paths
LOG_DIR = "logs"
MODEL_DIR = "models"
VIDEO_DIR = "videos"

# Evaluation / rendering
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
RENDER_CAMERA = "track"
RENDER_FRAME_SKIP = 2  # control runs at 50 Hz; render every 2nd step -> 25 fps video

# Scripted joystick demo: (label, [vx m/s, vy m/s, yaw rad/s], seconds)
# "stand still" comes last: once the policy settles into a stand with a zero command,
# a pure linear command does not restart the gait (a yaw command does).
DEMO_COMMANDS = [
    ("walk forward", [1.0, 0.0, 0.0], 4.0),
    ("run forward", [1.5, 0.0, 0.0], 3.0),
    ("turn left", [0.5, 0.0, 1.0], 3.0),
    ("turn right", [0.5, 0.0, -1.0], 3.0),
    ("strafe left", [0.0, 0.6, 0.0], 3.0),
    ("walk backward", [-0.8, 0.0, 0.0], 3.0),
    ("spin in place", [0.0, 0.0, 1.2], 3.0),
    ("stand still", [0.0, 0.0, 0.0], 2.0),
]
