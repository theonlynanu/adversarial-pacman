import time, sys, pathlib
from gym_wrapper.env import PacmanEnv
from gym_wrapper.our_agents import QPacman, QPacmanRelative
import argparse

DEFAULT_LAYOUT = "originalClassic"

def parse_cli():
    parser = argparse.ArgumentParser(
        description="Replay a game using a trained policy"
    )
    
    parser.add_argument("policy_path",
                        type=pathlib.Path,
                        help="Path to the .gz file with the given policy"
    )
    parser.add_argument("layout",
                        type=str,
                        help="Name of layout as found in layouts/ - does not include the .lay suffix"
    )
    parser.add_argument("agent_type",
                        type=str,
                        help="standard | relative - model used to train the policy - important to match this correctly!"
        
    )
    
    args = parser.parse_args()
    
    return args
    
    
args = parse_cli()

if args.agent_type == "standard":
    agentType = QPacman
elif args.agent_type == "relative":
    agentType = QPacmanRelative
else:
    print(f"Unknown agent type [{args.agent_type}] - use -h for options")
    sys.exit()
    
filepath = args.policy_path
layout = args.layout

pac_agent = agentType.load(
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