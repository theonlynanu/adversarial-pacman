import time
from pacman_engine.keyboard_agents import KeyboardAgent

from gym_wrapper.env import PacmanEnv
env = PacmanEnv(render_mode="graphics",
                layout="originalClassic",
               obs_type="condensed_grid",
               training_agent=None,
               pacman_agent=KeyboardAgent()
               )

obs, info = env.reset(seed=1234)

while True:
    obs, reward, terminal, truncated, info = env.step(0)
    env.render()
    if terminal or truncated:
        print("Final native score", info["native score"])
        print("Final cumulative calculated reward", info["cumulative reward"])
        break
    time.sleep(0.05)
    
env.close()
