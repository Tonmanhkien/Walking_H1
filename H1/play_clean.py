import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("H1 play (click-select + keyboard)")
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--ckpt", type=str, default=os.path.join("ckpts", "unitree_h1", "model_final.pt"))
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=2.5)
parser.add_argument("--cmd_start", type=int, default=9)
parser.add_argument("--cmd_dim", type=int, default=4)

parser.add_argument(
    "--world",
    type=str,
    default="plane",
    choices=[
        "plane",
        "warehouse",
        "warehouse-forklifts",
        "warehouse-shelves",
        "full-warehouse",
        "hospital",
        "office",
        "obstacle-sparse",
        "obstacle-medium",
        "obstacle-dense",
    ],
)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import carb
import omni
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from isaacsim.core.utils.stage import get_current_stage
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

from env.h1_base_env_cfg import H1BaseEnvCfg
import env.sim_env as sim_env

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

def load_policy(env, ckpt_path: str, device: str):
    from train.ppo import PPOAgent

    agent = PPOAgent(
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        batch_size=1,
        device=device,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    agent.load_state_dict(ckpt, strict=True)
    agent.eval()
    return agent


def spawn_world(name: str):
    if name == "plane":
        return
    if name == "warehouse":
        sim_env.create_warehouse_env()
    elif name == "warehouse-forklifts":
        sim_env.create_warehouse_forklifts_env()
    elif name == "warehouse-shelves":
        sim_env.create_warehouse_shelves_env()
    elif name == "full-warehouse":
        sim_env.create_full_warehouse_env()
    elif name == "hospital":
        sim_env.create_hospital_env()
    elif name == "office":
        sim_env.create_office_env()
    elif name == "obstacle-sparse":
        sim_env.create_obstacle_sparse_env()
    elif name == "obstacle-medium":
        sim_env.create_obstacle_medium_env()
    elif name == "obstacle-dense":
        sim_env.create_obstacle_dense_env()
    else:
        raise ValueError(f"Unknown world: {name}")


class H1KeyboardPlayer:
    def __init__(self, env, cmd_dim: int):
        self.env = env
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.scene.num_envs
        self.cmd_dim = cmd_dim

        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None

        self._camera_local = torch.tensor([-2.5, 0.0, 0.8], device=self.device)
        self._create_camera()

        self.commands = torch.zeros(self.num_envs, self.cmd_dim, device=self.device)
        self._try_init_from_manager()

        self._setup_keyboard()

    def _try_init_from_manager(self):
        try:
            cmd0 = self.env.unwrapped.command_manager.get_command("base_velocity")
            n = min(cmd0.shape[1], self.cmd_dim)
            self.commands[:, :n] = cmd0[:, :n]
        except Exception:
            pass

    def _create_camera(self):
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")

        self.camera_path = "/World/Camera"
        self.persp_path = "/OmniverseKit_Persp"

        cam_prim = stage.DefinePrim(self.camera_path, "Camera")
        cam_prim.GetAttribute("focalLength").Set(8.5)

        coi = cam_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi or not coi.IsValid():
            cam_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))

        self.viewport.set_active_camera(self.persp_path)

    def _setup_keyboard(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

        T, R = 1.0, 0.5

        def vec(vx=0.0, vy=0.0, wz=0.0, extra=0.0):
            out = torch.zeros(self.cmd_dim, device=self.device)
            if self.cmd_dim > 0:
                out[0] = vx
            if self.cmd_dim > 1:
                out[1] = vy
            if self.cmd_dim > 2:
                out[2] = wz
            if self.cmd_dim > 3:
                out[3] = extra
            return out

        self._map = {
            "UP": vec(T, 0.0, 0.0, 0.0),
            "DOWN": vec(0.0, 0.0, 0.0, 0.0),
            "LEFT": vec(T, 0.0, 0.0, -R),
            "RIGHT": vec(T, 0.0, 0.0, R),
            "ZEROS": vec(0.0, 0.0, 0.0, 0.0),
        }

    def _on_keyboard_event(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._map:
                if self._selected_id is not None:
                    self.commands[self._selected_id] = self._map[event.input.name]
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.persp_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if self._selected_id is not None:
                self.commands[self._selected_id] = self._map["ZEROS"]

        return True

    def update_selection(self):
        self._previous_selected_id = self._selected_id
        paths = self._prim_selection.get_selected_prim_paths()

        if len(paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.persp_path)
        elif len(paths) > 1:
            print("Multiple prims selected. Please select one.")
        else:
            parts = paths[0].split("/")
            if len(parts) >= 4 and parts[3].startswith("env_"):
                self._selected_id = int(parts[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("Selected prim is not env_* robot.")

        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            try:
                self.env.unwrapped.command_manager.reset([self._previous_selected_id])
                self._try_init_from_manager()
            except Exception:
                pass

    def _update_camera(self):
        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]
        cam_pos = quat_apply(base_quat, self._camera_local) + base_pos

        state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(cam_pos[0].item(), cam_pos[1].item(), cam_pos[2].item())
        tgt = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        state.set_position_world(eye, True)
        state.set_target_world(tgt, True)


def main():
    device = getattr(args, "device", "cuda:0")

    env_cfg = H1BaseEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = args.env_spacing

    try:
        env_cfg.commands.base_velocity.resampling_time_range = (0.0, 0.0)
    except Exception:
        pass

    spawn_world(args.world)
    env = gym.make("Isaac-Velocity-Flat-H1-v0", cfg=env_cfg)
    env = IsaacWrapper(env)

    agent = load_policy(env, args.ckpt, device=device)
    player = H1KeyboardPlayer(env, cmd_dim=args.cmd_dim)

    obs = env.reset()
    cmd_slice = slice(args.cmd_start, args.cmd_start + args.cmd_dim)

    with torch.inference_mode():
        while simulation_app.is_running():

            player.update_selection()
            if obs.shape[1] >= args.cmd_start + args.cmd_dim:
                obs[:, cmd_slice] = player.commands

            action = agent.actor(obs)
            action = torch.clamp(action, -1.0, 1.0)

            obs, _, _, _, _ = env.step(action)


    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
