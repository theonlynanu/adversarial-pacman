import time
from gym_wrapper.env import PacmanEnv
from gym_wrapper.our_agents import QPacman

pac_agent = QPacman.load(
    "q_pacman.pkl.gz",
    epsilon = 0,
    decay_rate = 1.0
)

env = PacmanEnv(
    render_mode="graphics",
    obs_type="condensed_grid",
    training_agent=None,
    pacman_agent=pac_agent
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