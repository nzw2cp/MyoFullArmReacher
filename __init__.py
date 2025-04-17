import os
import numpy as np

from myosuite.utils import gym; register=gym.register
from myosuite.envs.env_variants import register_env_variant

curr_dir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')


# utility to register envs with all muscle conditions
def register_env_with_variants(id, entry_point, max_episode_steps, kwargs):
    # register_env_with_variants base env
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )
    #register variants env with sarcopenia  #TODO: needs to be tested with myochallenge models
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'sarcopenia'},
            variant_id=id[:3]+"Sarc"+id[3:],
            silent=True
        )
    #register variants with fatigue  #TODO: needs to be tested with myochallenge models
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'fatigue'},
            variant_id=id[:3]+"Fati"+id[3:],
            silent=True
        )


# MyoFullArmReacher ===================================================
register_env_with_variants(
    id='myoFullArmReacher-v0',
    entry_point="MyoFullArmReacher.reacher_v0:ReacherEnvV0",
    max_episode_steps=200,
    kwargs={
        'model_path': curr_dir+'/myoarm_reacher.xml',
        'normalize_act': True,
        'frame_skip': 5,
        'pos_th': 0.1,              # cover entire base of the receptacle
        'rot_th': np.inf,           # ignore rotation errors
        'target_xyz_range': {'high':[0.2, -.1, 0.9], 'low':[0.0, -.35, 0.9]},
        'target_rxryrz_range': {'high':[0.0, 0.0, 0.0], 'low':[0.0, 0.0, 0.0]}
    }
)


