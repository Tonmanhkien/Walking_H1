# play.py
import argparse
import os
import sys

parser = argparse.ArgumentParser("Play (keyboard inference) for H1 PPO checkpoint.")
parser.add_argument(
    "--ckpt",
    type=str,
    default=os.path.join("ckpts", "unitree_h1", "model_final.pt"),
    help="Path to torch checkpoint saved by train.py (agent.state_dict()).",
)

parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--cmd_start", type=int, default=9, help="Start index of command in obs['policy'].")
parser.add_argument("--cmd_dim", type=int, default=4, help="Command dimension stored in obs slice.")

# IsaacLab AppLauncher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim *before* importing omni/carb/isaacsim modules
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import carb
import omni
from isaacsim.core.utils.stage import get_current_stage
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

from isaaclab.utils.math import quat_apply

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from train.ppo import PPOAgent
from env.h1_env import H1RoughEnvCfg_PLAY
from env.sim_env import create_warehouse_env, create_warehouse_forklifts_env, create_warehouse_shelves_env, create_full_warehouse_env, create_hospital_env, create_office_env



# --------------------------
# Helpers / Classes
# --------------------------
class IsaacWrapper(gym.Wrapper):
    """Match your train.py wrapper: obs = obs['policy']."""
    def __init__(self, env):
        super().__init__(env)
        self.num_obs = env.observation_space["policy"].shape[1]
        self.num_actions = env.action_space.shape[1]

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs["policy"]

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        return obs["policy"], r, terminated, truncated, info


class H1KeyboardPlayer:
    """Click-select robot env_i and control commands with arrow keys + camera follow."""
    def __init__(self, env, num_envs: int, cmd_dim: int):
        self.env = env
        self.device = env.unwrapped.device

        self.num_envs = num_envs
        self.cmd_dim = cmd_dim

        # selection state
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None

        # camera setup
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)
        self._create_camera()

        # commands buffer
        self.commands = torch.zeros(self.num_envs, self.cmd_dim, device=self.device)

        # init commands from command manager if available
        try:
            cmd0 = self.env.unwrapped.command_manager.get_command("base_velocity")
            n = min(cmd0.shape[1], self.cmd_dim)
            self.commands[:, :n] = cmd0[:, :n]
        except Exception:
            pass

        self._setup_keyboard()

    def _create_camera(self):
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")

        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"

        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)

        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))

        self.viewport.set_active_camera(self.perspective_path)

    def _setup_keyboard(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

        T = 1.0
        R = 0.5

        def cmd(vx=0.0, vy=0.0, wz=0.0, extra=0.0):
            out = torch.zeros(self.cmd_dim, device=self.device)
            if self.cmd_dim > 0: out[0] = vx
            if self.cmd_dim > 1: out[1] = vy
            if self.cmd_dim > 2: out[2] = wz
            if self.cmd_dim > 3: out[3] = extra
            return out

        self._key_to_control = {
            "UP": cmd(T, 0.0, 0.0, 0.0),
            "DOWN": cmd(0.0, 0.0, 0.0, 0.0),
            "LEFT": cmd(T, 0.0, 0.0, -R),
            "RIGHT": cmd(T, 0.0, 0.0, R),
            "ZEROS": cmd(0.0, 0.0, 0.0, 0.0),
        }

    def _on_keyboard_event(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_to_control:
                # IMPORTANT: allow selected_id==0
                if self._selected_id is not None:
                    self.commands[self._selected_id] = self._key_to_control[event.input.name]
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if self._selected_id is not None:
                self.commands[self._selected_id] = self._key_to_control["ZEROS"]

    def update_selected_object(self):
        self._previous_selected_id = self._selected_id

        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims selected. Please select only one.")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3].startswith("env_"):
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("Selected prim is not an H1 robot env_*")

        # reset command manager for old env (optional)
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            try:
                self.env.unwrapped.command_manager.reset([self._previous_selected_id])
                cmd0 = self.env.unwrapped.command_manager.get_command("base_velocity")
                n = min(cmd0.shape[1], self.cmd_dim)
                self.commands[:, :n] = cmd0[:, :n]
            except Exception:
                pass

    def _update_camera(self):
        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


# --------------------------
# main(): now only orchestration
# --------------------------
def main():
    # Build env
    env_cfg = H1RoughEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 1_000_000.0

    env = gym.make("Isaac-Velocity-Rough-H1-v0")
    env = IsaacWrapper(env)

    # Load agent
    agent = PPOAgent(
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        batch_size=1,
        device=args.device,
    )
    ckpt = torch.load(args.ckpt, map_location=args.device)
    agent.load_state_dict(ckpt)
    agent.eval()
    print(f"[INFO] Loaded checkpoint: {args.ckpt}")

    player = H1KeyboardPlayer(env, num_envs=args.num_envs, cmd_dim=args.cmd_dim)

    obs = env.reset()
    cmd_slice = slice(args.cmd_start, args.cmd_start + args.cmd_dim)

    with torch.inference_mode():
        while simulation_app.is_running():
            player.update_selected_object()

            # overwrite command in obs BEFORE inference (like IsaacLab demo)
            if obs.shape[1] >= args.cmd_start + args.cmd_dim:
                obs[:, cmd_slice] = player.commands

            action = agent.actor(obs)
            action = torch.clamp(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated | truncated
            if torch.any(done):
                pass

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
