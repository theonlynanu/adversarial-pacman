import time

from gym_wrapper.env import PacmanEnv
env = PacmanEnv(render_mode="graphics",
               obs_type="grid",
               training_agent=None,
               )

obs, info = env.reset(seed=1234)

while True:
    obs, reward, terminal, truncated, info = env.step(0)
    env.render()
    if terminal or truncated:
        print("Final score", info["score"])
        break
    time.sleep(0.05)
    
env.close()
