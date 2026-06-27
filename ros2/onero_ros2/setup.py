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
        (os.path.join('share', package_name, 'description'), glob('description/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'onero-sdk'],
    zip_safe=True,
    maintainer='Onero Robotics',
    maintainer_email='dev@onerobot.com',
    description='ROS2 Control hardware interface for Onero robot arms',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [],
    },
)
