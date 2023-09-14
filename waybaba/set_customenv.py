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

OPEN55 = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#OOOOO#\\"+\
        "#######"

register(
    id='maze2d-openlarge-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec':OPEN_LARGE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 6.7,
        'ref_max_score': 273.99,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
    }
)


register(
    id='maze2d-open55-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=10000, # ! the origin value is 150
    kwargs={
        'maze_spec':OPEN55,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 0.01,
        'ref_max_score': 20.66,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-sparse.hdf5'
    }
)
