from gym.envs.registration import register 
register(id='opt-env-v0', entry_point='final_project.envs:OptEnv',)
register(id='opt-env-v1', entry_point='final_project.envs:OptEnv1',)