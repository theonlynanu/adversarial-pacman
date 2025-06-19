import random, tqdm
from gym_wrapper.env import PacmanEnv
from gym_wrapper.our_agents import QPacman


N_EPISODES = 300_000
MAX_STEPS = 10_000

agent = QPacman(gamma = 0.99, epsilon=1)
env = PacmanEnv(
    render_mode=None,
    obs_type="condensed_grid",
    training_agent="pacman"
)

_action_to_idx = {d: i for i, d in enumerate(env._actions)}

for episode in tqdm.trange(N_EPISODES, desc="Q Training"):
    obs, _ = env.reset()
    state_prev = env.state
    done = False
    steps = 0
    
    while not done and steps < MAX_STEPS:
        dir_action = agent.get_action(state_prev)
        idx_action = _action_to_idx[dir_action]
        
        obs, reward, terminated, truncated, info = env.step(idx_action)
        done = terminated or truncated
        state_next = env.state
        
        agent.observe_transition(state_prev, dir_action, state_next, reward)
        
        state_prev = state_next
        steps += 1
        
    
env.close()
agent.save("q_pacman.pkl.gz")
print("Training Finished, tables saved.")