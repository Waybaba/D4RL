import gym
import logging
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, indicator=[".gitignore"])
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import pickle
import gzip
import h5py
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, a, tgt, done, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(0.0)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())

def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def plot_and_save_waypoints(obs, target, maze_arr, file_name):
    """
    obs: list of observations # N,4
    target: target position # 2
    maze_arr: maze array # H,W, with 11 means empty and 10 means wall
    """
    waypoints = np.array(obs)
    h, w = maze_arr.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, h-0.5)
    ax.set_ylim(-0.5, w-0.5)


    for i in range(h):
        for j in range(w):
            if maze_arr[i, j] == 10:
                ax.fill([i-0.5, i+0.5, i+0.5, i-0.5], [j-0.5, j-0.5, j+0.5, j+0.5], color='gray')
                
    ax.plot(waypoints[:, 0], waypoints[:, 1], 'ro-', zorder=0)

    # Annotate start point
    ax.scatter(waypoints[0, 0], waypoints[0, 1], color='blue', edgecolors='black', s=100)
    ax.annotate('Start', (waypoints[0, 0], waypoints[0, 1]), textcoords="offset points", xytext=(-10,-10), ha='center')

    # Annotate target point
    ax.scatter(target[0], target[1], color='green', marker='*', edgecolors='black', s=200)
    ax.annotate('End', (target[0], target[1]), textcoords="offset points", xytext=(-10,-10), ha='center')


    # ax.axis('off')  # Turn off axes
    ax.axis('tight')  # Set limits to smallest possible
    fig.tight_layout(pad=0)  # Remove padding
    fig.savefig(file_name)



def main():
    CUSTOM_TARGET = {
        "tl2br": {
            "location": (1.0, 1.0),
            "target": np.array([7, 10]),
        },
        "br2tl": {
            "location": (7.0, 10.0),
            "target": np.array([1, 1]),
        },
        "tr2bl": {
            "location": (7.0, 1.0),
            "target": np.array([1, 10]),
        },
        "bl2tr": {
            "location": (1.0, 10.0),
            "target": np.array([7, 1]),
        },
        "2wayAv1": {
            "location": (1.0, 1.0),
            "target": np.array([3, 4]),
        },
        "2wayAv2": {
            "location": (3.0, 4.0),
            "target": np.array([1, 1]),
        },
        "2wayBv1": {
            "location": (5, 6),
            "target": np.array([1, 10]),
        },
        "2wayBv2": {
            "location": (1, 10.0),
            "target": np.array([5, 6]),
        },
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1', help='Maze type')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    parser.add_argument('--custom_target', type=str, default=None, help='Custom target')
    parser.add_argument('--custom_target_ratio', type=float, default=1.0, help='Custom target ratio')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    controller = waypoint_controller.WaypointController(maze)
    env = maze_model.MazeEnv(maze)

    # recurrently create ./debug/data/waypoints-<s_idx>.png, clear the content of ./debug/data first
    if not os.path.exists('debug/data'): 
        os.makedirs('debug/data')
    else:
        os.system('rm -rf debug/data/*')

    env.set_target()
    
    if args.custom_target is not None:
        # env.set_state(self.CUSTOM_TARGET[cfg.trainer.custom_target]["state"])
        env.set_target(CUSTOM_TARGET[args.custom_target]["target"])
        s = env.reset_to_location(CUSTOM_TARGET[args.custom_target]["location"])
        print("###")
        print(f"use custom target ### {args.custom_target} ###")
        print("state", env.state_vector())
        print("target", env._target)
    else:
        s = env.reset()
    act = env.action_space.sample()
    done = False

    data = reset_data()
    ts = eps_start = 0
    for s_idx in tqdm(range(args.num_samples)):
        position = s[0:2]
        velocity = s[2:4]
        act, done = controller.get_action(position, velocity, env._target)
        if args.noisy:
            act = act + np.random.randn(*act.shape)*0.5

        act = np.clip(act, -1.0, 1.0)
        if ts >= max_episode_steps:
            done = True
        append_data(data, s, act, env._target, done, env.sim.data)

        ns, _, _, _ = env.step(act)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1
        if done:
            done = False
            if s_idx < 100000:
                plot_and_save_waypoints(data["observations"][eps_start:], env._target, env.maze_arr, 'debug/data/waypoints-%d.png' % s_idx)
                print('save to path: ./debug/data/waypoints-%d.png' % s_idx)
            if args.custom_target is not None and np.random.rand() < args.custom_target_ratio:
                # env.set_state(self.CUSTOM_TARGET[cfg.trainer.custom_target]["state"])
                env.set_target(CUSTOM_TARGET[args.custom_target]["target"])
                env.reset_to_location(CUSTOM_TARGET[args.custom_target]["location"])
                print("###")
                print(f"use custom target ### {args.custom_target} ###")
                print("state", env.state_vector())
                print("target", env._target)
            else:
                env.set_target()
            eps_start = s_idx
            ts = 0
        else:
            s = ns

        if args.render:
            # env.render()
            pass

    
    if args.noisy:
        fname = '%s-noisy.hdf5' % args.env_name
    else:
        fname = '%s.hdf5' % args.env_name
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == "__main__":
    main()
