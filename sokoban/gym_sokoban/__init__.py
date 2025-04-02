# Core Library
import logging

# Third party
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="Sokoban-v0", 
    entry_point="gym_sokoban.envs:SokobanEnv",
    kwargs={"difficulty": "unfiltered"},
)

register(
    id="Sokoban-easy-v0", 
    entry_point="gym_sokoban.envs:SokobanEnv",
    kwargs={"difficulty": "easy"},
)

register(
    id="Sokoban-medium-v0",
    entry_point="gym_sokoban.envs:SokobanEnv",
    kwargs={"difficulty": "medium"},
)
register(
    id="Sokoban-hard-v0",
    entry_point="gym_sokoban.envs:SokobanEnv",
    kwargs={"difficulty": "hard"},
)
register(
    id="Sokoban-test-v0",
    entry_point="gym_sokoban.envs:SokobanEnv",
    kwargs={"difficulty": "test"},
)

register(
        id="Sokoban-valid-v0",
        entry_point="gym_sokoban.envs:SokobanEnv",
        kwargs={"difficulty": "valid"},
)

register(
        id="Sokoban-probing-v0",
        entry_point="gym_sokoban.envs:SokobanEnv",
        kwargs={"difficulty": "probing"},
)

for exp_type, exp_ids in [
    ("med", range(1000)),
    ("cutoffpusht4", range(200)),
    ("shortcut", range(200)),
    ("boxshortcut", range(200)),
    ("cutoffpushc2", range(80)),
    ("cutoffpushc6", range(80)),
    ("cutoffpushc10", range(80)),
    ("cutoffpushc14", range(80)),
    ]:
    for exp_id in exp_ids:
        for mode in ["clean", "corrupt"]:
            register(
                id=f"Sokoban-{exp_type}_{mode}_{exp_id:04}-v0", 
                entry_point="gym_sokoban.envs:SokobanEnv",
                kwargs={"difficulty": f"exp_{exp_type}_{mode}_{exp_id:04}"},
            )
