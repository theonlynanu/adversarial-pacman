import random, tqdm, sys, pathlib
from gym_wrapper.env import PacmanEnv
from gym_wrapper.our_agents import QPacman

########## HYPER PARAMS ##########
GAMMA = 0.99
EPSILON_START = 1
EPSILON_MIN = 0.001
DECAY_RATE = 0.99999


N_EPISODES = 100_000
MAX_STEPS = 10_000
SAVE_CHECKPOINTS = True
CHECKPOINT_FREQUENCY = 5000
CONTINUING_TRAINING = False

def create_position_heatmap(position_log, title="Pac-Man Positional Visits"):
    xs, ys = zip(*position_log)
    w, h = max(xs) + 1, max(ys) + 1
    counts = np.zeros((h,w), dtype=int)
    for x, y in position_log:
        counts[y, x] += 1
        
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(counts, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    fig.colorbar(im, ax=ax, label="# visits")
    plt.tight_layout()
    plt.savefig("heatmap.png")

def confirm_retrain(filepath, is_present):
    if CONTINUING_TRAINING and is_present:
        print(f"Confirm that you are retraining {filepath}, updating its current state")
    elif CONTINUING_TRAINING and not is_present:
        print(f"{filepath} not present, confirm you are writing new policy to {filepath}")
    elif not CONTINUING_TRAINING and is_present:
        print(f"Confirm that you are OVERWRITING the policy at {filepath} - this will destroy the current version!")
    elif not CONTINUING_TRAINING and not is_present:
        print(f"Writing new policy to {filepath}...")
        return
    
    confirmed = False
    while not confirmed:
        confirmation = input("[(Y)es / (N)o]    > ").lower()
        if confirmation == 'y' or confirmation == "yes":
            return
        elif confirmation == 'n' or confirmation == "no":
            sys.exit()
        else:
            print(f"Response {confirmation} not understood.")

def create_agent(table_path):
    if CONTINUING_TRAINING:
        return QPacman.load(table_path, gamma = GAMMA, epsilon=EPSILON_START, epsilon_min=EPSILON_MIN, decay_rate=DECAY_RATE)
    else:
        return QPacman(gamma = GAMMA, epsilon=EPSILON_START, epsilon_min=EPSILON_MIN, decay_rate=DECAY_RATE)


if len(sys.argv) > 1:
    table_path = pathlib.Path(sys.argv[1])
else:
    table_path = pathlib.Path("q_pacman.pkl.gz")

is_present = table_path.exists()

confirm_retrain(table_path, is_present)

agent = create_agent(table_path)

env = PacmanEnv(
    render_mode=None,
    obs_type="condensed_grid",
    training_agent="pacman",
    layout="originalClassic"
)

_action_to_idx = {d: i for i, d in enumerate(env._actions)}

wins = 0
losses = 0
positions = []

for episode in tqdm.trange(N_EPISODES, desc="Q Training"):
    obs, _ = env.reset()
    state_prev = env.state
    done = False
    steps = 0
    
    if SAVE_CHECKPOINTS:
        if episode % CHECKPOINT_FREQUENCY == 0:
            agent.save(f"checkpoints/q_ep{episode}.pkl.gz")
    
    while not done and steps < MAX_STEPS:
        dir_action = agent.get_action(state_prev)
        idx_action = _action_to_idx[dir_action]
        
        positions.append(env.state.get_pacman_position())
        
        obs, reward, terminated, truncated, info = env.step(idx_action)
        done = terminated or truncated
        if done:
            if env.state.is_win():
                wins += 1
            elif env.state.is_lose():
                losses += 1
            else:
                print("ERROR - ended early")        # Shouldn't really hit this unless we get stuck in a weird loop
            
        state_next = env.state
        
        agent.observe_transition(state_prev, dir_action, state_next, reward)
        
        state_prev = state_next
        steps += 1
    agent.decay_epsilon()
    
        
print("Final epsilon ", agent.epsilon)
    
env.close()
agent.save("q_pacman.pkl.gz")

import matplotlib.pyplot as plt
import numpy as np

visits = np.array(list(agent.visited.values()))
plt.hist(visits, bins=50, log=True)
plt.xlabel("# visits per state"); plt.ylabel("count"); plt.title("State-visit histogram")
plt.savefig("state_visit_histogram.png")

create_position_heatmap(positions)

print("Training Finished, tables saved.")
print("SUMMARY STATISTICS:")
print("Q-Table size: ", len(agent.Q))
print("States seen: ", len(agent.visited))
print("Avg actions per seen state: ", len(agent.Q) / len(agent.visited))
print("Wins: ", wins)
print("Losses: ", losses)