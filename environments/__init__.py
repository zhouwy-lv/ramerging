from gym.envs.registration import register
from . import RampMerging_v0,RampMergeSafe_v1
register(
    id='RampMerging-v0',
    entry_point='environments.RampMerging_v0:RampMergingEnv',
    trials = 10,
    # reward_threshold = 200.,
)

register(
    id='RampMerging-v1',
    entry_point='environments.RampMergeSafe_v1:RampMergingEnv',
    trials = 10,
    # reward_threshold = 200.,
)
# 
# register(
#     id='ObstacleAvoidance-v1',
#     entry_point='environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1_two_eyes:ObstacleAvoidanceScenarioTwoEyes',
#     trials = 10,
#     reward_threshold = 200.,
# )
# 
# register(
#     id='ObstacleAvoidance-v2',
#     entry_point='environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1_dynamic:ObstacleAvoidanceScenarioDynamic',
#     trials = 10,
#     reward_threshold = 200.,
# )
# 
# """single eye static"""
# register(
#     id='ObstacleAvoidance-v3',
#     entry_point='environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1_single_eye:ObstacleAvoidanceScenarioSingleEye',
#     trials = 10,
#     reward_threshold = 200.,
# )
# 
# """single eye dynamic"""
# register(
#     id='ObstacleAvoidance-v31',
#     entry_point='environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1_single_eye:ObstacleAvoidanceScenarioSingleEyeDynamic',
#     trials = 10,
#     reward_threshold = 200.,
# )
# 
# 
# """时间序列环境：静态"""
# register(
#     id='ObstacleAvoidance-v4',
#     entry_point='environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1_time_seq:ObstacleAvoidanceScenarioTimeSeq',
#     trials = 10,
#     reward_threshold = 200.,
# )
# 
# """时间序列环境：动态"""
# register(
#     id='ObstacleAvoidance-v5',
#     entry_point='environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1_time_seq:ObstacleAvoidanceScenarioTimeSeqDynamic',
#     trials = 10,
#     reward_threshold = 200.,
# )