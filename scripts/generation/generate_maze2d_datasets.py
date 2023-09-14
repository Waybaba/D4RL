import gym
import logging
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, indicator=[".gitignore"])
import waybaba.set_customenv
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
        "r2l": {
            "location": (1.0, 2.0),
            "target": np.array([5, 4]),
        },
        "l2r": {
            "location": (5.0, 4.0),
            "target": np.array([1.0, 2.0]),
        },
    }
    TARGET_SET = {
        "4edge": [
            np.array([1.,3.]),
            np.array([5.,3.]),
            np.array([3.,1.]),
            np.array([3.,5.]),
        ],
        "4edgex": [
            np.array([1.,4.]),
            np.array([2.,1.]),
            np.array([5.,2.]),
            np.array([4.,5.]),
        ],
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
    if not os.path.exists('output/debug/data'): 
        os.makedirs('output/debug/data')
    else:
        os.system('rm -rf output/debug/data/*')
        os.system('rm -rf output/debug/images/*')

    env.set_target()
    
    if args.custom_target is not None:
        # env.set_state(self.CUSTOM_TARGET[cfg.trainer.custom_target]["state"])
        if args.custom_target == "onlytwo":
            # if close to l2r, set target as r2l, else l2r
            dis2l2r = np.linalg.norm(env.get_target() - CUSTOM_TARGET["l2r"]["location"])
            dis2r2l = np.linalg.norm(env.get_target() - CUSTOM_TARGET["r2l"]["location"])
            if dis2l2r < dis2r2l:
                env.set_target(CUSTOM_TARGET["r2l"]["target"])
            else:
                env.set_target(CUSTOM_TARGET["l2r"]["target"])
            s = env.reset()
        elif args.custom_target.startswith("sets"):
            """ sets%{set_name}
            e.g. sets%4edge
            each time, select new target from sets randomly (except the current one)
            """
            _, set_name = args.custom_target.split("%")
            sets = TARGET_SET[set_name]
            distances = [
                np.linalg.norm(env.get_target() - sets[i])
                for i in range(len(sets))
            ]
            idx_min = np.argmin(distances)
            # random select from others
            idx = np.random.choice([i for i in range(len(sets)) if i != idx_min])
            env.set_target(sets[idx])
            s = env.reset()
        else: 
            raise NotImplementedError
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
            if s_idx < 10000:
            # if False:
                if not os.path.exists('output/images'): 
                    os.makedirs('output/images')
                plot_and_save_waypoints(data["observations"][eps_start:], env._target, env.maze_arr, './output/debug/images/waypoints-%d.png' % s_idx)
                print('save to path: ./output/debug/images/waypoints-%d.png' % s_idx)
            if args.custom_target is not None and np.random.rand() < args.custom_target_ratio:
                if args.custom_target == "onlytwo":
                    # if close to l2r, set target as r2l, else l2r
                    dis2l2r = np.linalg.norm(env.get_target() - CUSTOM_TARGET["l2r"]["location"])
                    dis2r2l = np.linalg.norm(env.get_target() - CUSTOM_TARGET["r2l"]["location"])
                    if dis2l2r > dis2r2l:
                        env.set_target(CUSTOM_TARGET["r2l"]["target"])
                    else:
                        env.set_target(CUSTOM_TARGET["l2r"]["target"])
                elif args.custom_target.startswith("sets"):
                    """ sets%{set_name}
                    e.g. sets%4edge
                    each time, select new target from sets randomly (except the current one)
                    """
                    _, set_name = args.custom_target.split("%")
                    sets = TARGET_SET[set_name]
                    distances = [
                        np.linalg.norm(env.get_target() - sets[i])
                        for i in range(len(sets))
                    ]
                    idx_min = np.argmin(distances)
                    # random select from others
                    idx = np.random.choice([i for i in range(len(sets)) if i != idx_min])
                    env.set_target(sets[idx])
                    # s = env.reset()
                else:
                    raise NotImplementedError
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

    
    # folder $UDATADIR/models/diffuser/d4rl_dataset
    fdname = os.path.join(os.environ['UDATADIR'], 'models/diffuser/d4rl_dataset')
    if not os.path.exists(fdname): os.makedirs(fdname)

    if args.noisy:
        fname = os.path.join(fdname, '%s-%s-noisy.hdf5' % (args.env_name, str(args.num_samples)+("" if args.custom_target is None else "-"+args.custom_target)))
    else:
        fname = os.path.join(fdname, '%s-%s.hdf5' % (args.env_name, str(args.num_samples)+("" if args.custom_target is None else "-"+args.custom_target)))
    dataset = h5py.File(fname, 'w')

    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')

    # make a alias for the latest one
    os.system('rm -rf %s' % os.path.join(fdname, 'latest.hdf5'))
    os.system('ln -s %s %s' % (fname, os.path.join(fdname, 'latest.hdf5')))
    print('save to path: %s' % fname)

if __name__ == "__main__":
    main()
