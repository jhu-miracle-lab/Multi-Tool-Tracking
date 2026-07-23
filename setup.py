from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'surgical_tool_tracking_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=['tracking', 'tracking.*', 'core', 'core.*', 'ros2_node', 'ros2_node.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'Parameter'),
            glob('Parameter/*')),
        # Hand-eye calibration YAMLs, if any are checked in at the package
        # root (see tracking/arm_configs.py's build_arm_configs() fallback
        # search) -- glob() is fine with zero matches, unlike a literal
        # missing filename in data_files.
        (os.path.join('share', package_name),
            glob('pnp_*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joec',
    maintainer_email='yaochen_0719@outlook.com',
    description='Dual-arm surgical tool tracking ROS2 node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracking_node = ros2_node.tracking_node:main',
            'offline_node = ros2_node.offline_node:main',
            'dvrk_stereo_publisher = ros2_node.dvrk_stereo_publisher:main',
        ],
    },
)
