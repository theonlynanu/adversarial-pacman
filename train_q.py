import random, tqdm
from gym_wrapper.env import PacmanEnv
from gym_wrapper.our_agents import QPacman


N_EPISODES = 50_000
MAX_STEPS = 10_000

agent = QPacman(gamma = 0.99, epsilon=1, epsilon_min=0.005)
env = PacmanEnv(
    render_mode=None,
    obs_type="condensed_grid",
    training_agent="pacman"
)

_action_to_idx = {d: i for i, d in enumerate(env._actions)}

wins = 0
losses = 0

for episode in tqdm.trange(N_EPISODES, desc="Q Training"):
    obs, _ = env.reset()
    state_prev = env.state
    done = False
    steps = 0
    
    if episode % 500 == 0:
        agent.save(f"checkpoints/q_ep{episode}.pkl.gz")
    
    while not done and steps < MAX_STEPS:
        dir_action = agent.get_action(state_prev)
        idx_action = _action_to_idx[dir_action]
        
        obs, reward, terminated, truncated, info = env.step(idx_action)
        done = terminated or truncated
        if done:
            if env.state.is_win():
                wins += 1
            elif env.state.is_lose():
                losses += 1
            else:
                print("ERROR")
            
        state_next = env.state
        
        agent.observe_transition(state_prev, dir_action, state_next, reward)
        
        state_prev = state_next
        steps += 1
        
    
env.close()
agent.save("q_pacman.pkl.gz")

import matplotlib.pyplot as plt
import numpy as np

visits = np.array(list(agent.visited.values()))
plt.hist(visits, bins=50, log=True)
plt.xlabel("# visits per state"); plt.ylabel("count"); plt.title("State-visit histogram")
plt.show()

print("Training Finished, tables saved.")
print("SUMMARY STATISTICS:")
print("Q-Table size: ", len(agent.Q))
print("States seen: ", len(agent.visited))
print("Avg actions per seen state: ", len(agent.Q) / len(agent.visited))
print("Wins: ", wins)
print("Losses: ", losses)