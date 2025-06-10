from pathlib import Path
from setuptools import setup, find_packages
setup(
    name="adversarial-pacman",
    version="0.1.0",
    description="Pack Man.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29",
        "numpy>=1.24"
    ],
    entry_points={
        "gymnasium.envs": [
            "Pacman-v0 = gym_wrapper.env:PacmanEnv"
        ]
    },
    include_package_data=True,
)