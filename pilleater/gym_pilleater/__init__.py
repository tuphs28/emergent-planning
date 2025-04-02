from gym.envs.registration import register

register(
    id='gym_pilleater/PillEater-v0',
    entry_point='gym_pilleater.envs:PillEaterEnv',
)