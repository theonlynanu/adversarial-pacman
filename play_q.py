import time, sys, pathlib
from gym_wrapper.env import PacmanEnv
from gym_wrapper.our_agents import QPacman

DEFAULT_LAYOUT = "originalClassic"


if len(sys.argv) > 1:
    filepath = pathlib.Path(sys.argv[1])
    if len(sys.argv) > 2:
        layout = sys.argv[2]
    else:
        layout = DEFAULT_LAYOUT
else:
    filepath = pathlib.Path("q_pacman.pkl.gz")
    layout = DEFAULT_LAYOUT
    
if not filepath.exists():
    print(f"Policy file not found! - {filepath.name}")
    sys.exit()
else:
    print(f"Replaying using policy at {filepath.name}")

pac_agent = QPacman.load(
    filepath,
    epsilon = 0,
    epsilon_min = 0,
    decay_rate = 0,
)

env = PacmanEnv(
    render_mode="graphics",
    obs_type="condensed_grid",
    training_agent=None,
    pacman_agent=pac_agent,
    layout=layout
)

obs, _ = env.reset()

done = False

ACTION_TO_IDX = {d: i for i, d in enumerate(env._actions)}

while not done:
    action_dir = pac_agent.get_action(env.state)
    action_idx = ACTION_TO_IDX[action_dir]
    
    obs, reward, term, trunc, info = env.step(action_idx)
    
    done = term or trunc
    
    time.sleep(0.05)