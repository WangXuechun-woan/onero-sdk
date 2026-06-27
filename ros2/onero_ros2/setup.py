from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'onero_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'description'), glob('description/*.xacro')),
        (os.path.join('share', package_name, 'description', 'meshes', 'a1_r'),
         glob('description/meshes/a1_r/*.STL')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Onero Robotics',
    maintainer_email='dev@onerobot.com',
    description='ROS2 driver for Onero robot arms (standalone node + MoveIt2 compatible)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'arm_node = onero_ros2.arm_node:main',
        ],
    },
)
