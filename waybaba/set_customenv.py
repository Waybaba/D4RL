from d4rl.pointmaze.gridcraft.maze_model import MazeEnv, OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE, U_MAZE_EVAL, MEDIUM_MAZE_EVAL, LARGE_MAZE_EVAL
from gym.envs.registration import register


OPEN_LARGE = \
        "############\\"+\
        "#OOOOOOOOOO#\\"+\
        "#OOOOOOOOOO#\\"+\
        "#OOOOOOOOOO#\\"+\
        "#OOOOOGOOOO#\\"+\
        "#OOOOOOOOOO#\\"+\
        "#OOOOOOOOOO#\\"+\
        "#OOOOOOOOOO#\\"+\
        "############"

register(
    id='maze2d-open-large',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=50,
    kwargs={
        'maze_spec':OPEN_LARGE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 6.7,
        'ref_max_score': 273.99,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
    }
)