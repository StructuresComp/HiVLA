from setuptools import setup, find_packages

setup(
    name="hivla_project",
    version="0.1.1",
    description="Official Reinforcement Learning implementation for the paper 'Your Vision-Language-Action Model Already Has Attention Heads For Path Deviation Detection'",
    keywords=["isaac_lab", "rl", "navigation"],
    packages=find_packages("."),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "isaaclab",  # Isaac Lab Core
        "skrl==1.4.3",  # RL Library
    ],
)