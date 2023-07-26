import gym
import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import pickle
import gzip
import h5py
import argparse
import matplotlib.pyplot as plt
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
    maze_arr: maze array # H,W, with 10 means emtpy and 11 means wall
    """
    waypoints = np.array(obs)
    plt.figure()
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'ro-')

    # Plot the start point
    plt.scatter(waypoints[0, 0], waypoints[0, 1], color='blue', edgecolors='black', s=100) 

    # Plot the target
    plt.scatter(target[0], target[1], color='green', marker='*', edgecolors='black', s=200) 

    plt.title('Waypoints')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.savefig(file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1', help='Maze type')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
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
    s = env.reset()
    act = env.action_space.sample()
    done = False

    data = reset_data()
    ts = eps_start = 0
    for s_idx in range(args.num_samples):
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
            env.set_target()
            done = False
            
            plot_and_save_waypoints(data["observations"][eps_start:], env._target, env.maze_arr, 'debug/data/waypoints-%d.png' % s_idx)
            print('waypoints-%d.png saved' % s_idx)
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
