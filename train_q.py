import random, tqdm, sys, pathlib
from gym_wrapper.env import PacmanEnv
from gym_wrapper.our_agents import QPacman, QPacmanRelative
import matplotlib.pyplot as plt
import numpy as np
import argparse

########## HYPER PARAMS ##########
GAMMA = 0.99
EPSILON_START = 1
EPSILON_MIN = 0.02
DECAY_RATE = 0.9998

# ENSURE THAT THE CURRENT LAYOUT IS IN-LINE WITH WHAT YOU WANT TO TRAIN!
LAYOUT = "mediumClassic"

N_EPISODES = 50_000
MAX_STEPS = 2_000
SAVE_CHECKPOINTS = False
CHECKPOINT_FREQUENCY = 5000
CHECKPOINT_PREFIX = "checkpoints/curriculum_2"
CONTINUING_TRAINING = False

def create_position_heatmap(position_log: np.ndarray, title="Pac-Man Positional Visits", filename=f"heatmaps/{LAYOUT}_{N_EPISODES}_heatmap.png"):
    if position_log.ndim != 2:
        raise ValueError("Position log must be a 2-dimensional array (HxW)!")
        
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(position_log, origin="lower", cmap="hot", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="# visits")
    cbar.ax.yaxis.set_label_position("left")
    
    plt.tight_layout()
    fig_path = pathlib.Path(filename)
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    
    return fig_path

def confirm_retrain(filepath, is_present):
    if CONTINUING_TRAINING and is_present:
        print(f"Confirm that you are retraining {filepath} using layout {LAYOUT}, updating its current state")
    elif CONTINUING_TRAINING and not is_present:
        print(f"{filepath} not present, confirm you are writing new policy to {filepath} using {LAYOUT}")
    elif not CONTINUING_TRAINING and is_present:
        print(f"Confirm that you are OVERWRITING the policy at {filepath} on layout {LAYOUT}- this will destroy the current version!")
    elif not CONTINUING_TRAINING and not is_present:
        print(f"Writing new policy to {filepath} using {LAYOUT}...")
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



def create_agent(table_path, agentType):
    if CONTINUING_TRAINING:
        return agentType.load(table_path, gamma = GAMMA, epsilon=EPSILON_START, epsilon_min=EPSILON_MIN, decay_rate=DECAY_RATE)
    else:
        return agentType(gamma = GAMMA, epsilon=EPSILON_START, epsilon_min=EPSILON_MIN, decay_rate=DECAY_RATE)
    
def parse_cli() -> tuple[pathlib.Path, str, type]:
    parser = argparse.ArgumentParser(
        description="Train a Q-learning Pac-Man Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Positional Arguments
    parser.add_argument(
        "policy_filepath",
        type=pathlib.Path,
        help="Path to the .gz file from which the Q-table will be loaded from and/or saved"
    )
    
    parser.add_argument(
        "layout_name",
        type=str,
        help="Name of layout as found in layouts/ - does not include the .lay suffix"
    )
    
    parser.add_argument(
        "agent_type",
        type=str,
        help="standard | relative - whether to use the standard QPacman agent or the QPacmanRelative agent (relative is recommended)"
    )
    
    # Optional arguments
    
    # Solo flags
    parser.add_argument("-r", "--restart",
                        action="store_true",
                        help="Overwrite existing q table"
    )
    parser.add_argument("-s", "--save-checkpoints",                    
                        action="store_true",
                        help="Periodically store checkpoint files in checkpoints/"
    )
    
    # Hyper-parameter tuning
    parser.add_argument("--episodes", "-e",
                        type=int,
                        default=N_EPISODES,
                        help="Number of training episodes"                    
    )
    parser.add_argument("--epsilon-start",
                        type=float,
                        default=EPSILON_START,
                        help="Starting epsilon for epsilon-greedy decay"
    )
    parser.add_argument("--epsilon-min",
                        type=float,
                        default=EPSILON_MIN,
                        help="Floor for epsilon decay - prevent unecessarily long floating point comparisons"
    )
    parser.add_argument("-g", "--gamma",
                        type=float,
                        default=GAMMA,
                        help="Discount factor for future reward"                    
    )
    parser.add_argument("-d", '--decay',
                        type=float,
                        default=DECAY_RATE,
                        help="Per-episode decay rate"
    )
    parser.add_argument("--checkpoint-prefix",
                        type=str,
                        default=CHECKPOINT_PREFIX,
                        help="Prefix for each checkpoint policy, followed by '_[episode_number].gz'."
    )
    parser.add_argument("--checkpoint-freq",
                        type=int,
                        default=CHECKPOINT_FREQUENCY,
                        help="Frequency with which to save a checkpoint - will save every X episodes")
    
    args = parser.parse_args()
    
    agent_class = {"Standard": QPacman,
                   "Relative": QPacmanRelative}
    
    return args

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ PROGRAM START ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def main():
    global LAYOUT, CONTINUING_TRAINING, SAVE_CHECKPOINTS, N_EPISODES, EPSILON_START, EPSILON_MIN, GAMMA, DECAY_RATE, CHECKPOINT_PREFIX, CHECKPOINT_FREQUENCY
    
    
    args = parse_cli()
    if args.agent_type == "standard":
        agentType = QPacman
    elif args.agent_type == "relative":
        agentType = QPacmanRelative
    else:
        print(f"Unknown agent type [{args.agent_type}] - use -h for options")
        sys.exit()
    
    table_path = args.policy_filepath
    LAYOUT = args.layout_name
    CONTINUING_TRAINING = not args.restart
    SAVE_CHECKPOINTS = args.save_checkpoints
    N_EPISODES = args.episodes
    EPSILON_START = args.epsilon_start
    EPSILON_MIN = args.epsilon_min
    GAMMA = args.gamma
    DECAY_RATE = args.decay
    CHECKPOINT_PREFIX = args.checkpoint_prefix
    CHECKPOINT_FREQUENCY = args.checkpoint_freq

    is_present = table_path.exists()

    confirm_retrain(table_path, is_present)

    agent = create_agent(table_path, agentType)

    env = PacmanEnv(
        render_mode=None,
        obs_type="condensed_grid",
        training_agent="pacman",
        layout=LAYOUT
    )

    _action_to_idx = {d: i for i, d in enumerate(env._actions)}

    wins = 0
    losses = 0
    h, w = env.layout.height, env.layout.width
    pos_counts = np.zeros((h, w), dtype=np.uint32)

    early_exits = 0

    for episode in tqdm.trange(N_EPISODES, desc="Q Training"):
        obs, _ = env.reset()
        state_prev = env.state
        done = False
        steps = 0
        
        if SAVE_CHECKPOINTS:
            if episode % CHECKPOINT_FREQUENCY == 0:
                agent.save(f"{CHECKPOINT_PREFIX}_{episode}.gz")
        
        while not done and steps < MAX_STEPS:
            dir_action = agent.get_action(state_prev)
            idx_action = _action_to_idx[dir_action]
            
            x, y = map(int, env.state.get_pacman_position())
            pos_counts[y, x] += 1
            
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
            if steps == MAX_STEPS:
                early_exits += 1
                print(f"Did not complete game in {MAX_STEPS} steps")
            
            steps += 1
        agent.decay_epsilon()
        
            
    print("Final epsilon ", agent.epsilon)
        
    env.close()
    agent.save(table_path)


    visits = np.array(list(agent.visited_sa.values()))
    plt.hist(visits, bins=50, log=True)
    plt.xlabel("# visits per state"); plt.ylabel("count"); plt.title("State-visit histogram")
    plt.savefig("state_visit_histogram.png")

    create_position_heatmap(pos_counts)

    print("Training Finished, tables saved.")
    print("SUMMARY STATISTICS:")
    print("Q-Table size: ", len(agent.Q))
    states_seen = len({s for (s, _a) in agent.visited_sa})
    print("States seen: ", states_seen)
    print("Avg actions per seen state: ", len(agent.Q) / states_seen)
    print("Wins: ", wins)
    print("Losses: ", losses)
    print("Early quits: ", early_exits)
    
    
if __name__ == "__main__":
    main()