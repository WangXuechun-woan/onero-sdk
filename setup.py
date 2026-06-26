from setuptools import setup, find_packages

setup(
    name="onero-sdk",
    version="0.3.0",
    description="Python SDK for onero robotic arms",
    author="OneRobotics",
    python_requires=">=3.10",
    packages=find_packages(),
    package_data={
        "onero_sdk": ["urdf/**/*.urdf"],
    },
    include_package_data=True,
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "ik": ["pin>=2.7", "pink>=2.0"],
        "dynamics": ["pin>=2.7"],
        "sim": ["mujoco>=3.0"],
        "calibration": ["scipy>=1.10"],
        "trajectory": ["ruckig>=0.9", "toppra>=0.6"],
        "full": ["pin>=2.7", "pink>=2.0", "mujoco>=3.0", "scipy>=1.10",
                 "ruckig>=0.9", "toppra>=0.6"],
        "dev": ["pytest>=7.0"],
        "config": ["pyyaml>=6.0"],
    },
)
