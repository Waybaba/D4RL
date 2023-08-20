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
# LARGE_MAZE = \
#         "############\\"+\
#         "#OOOO#OOOOO#\\"+\
#         "#O##O#O#O#O#\\"+\
#         "#OOOOOO#OOO#\\"+\
#         "#O####O###O#\\"+\
#         "#OO#O#OOOOO#\\"+\
#         "##O#O#O#O###\\"+\
#         "#OO#OOO#OGO#\\"+\
#         "############"

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