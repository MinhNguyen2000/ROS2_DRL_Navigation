from setuptools import find_packages, setup

package_name = 'covariance_filter'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='matt',
    maintainer_email='mtidd2@unb.ca',
    description='Takes in ROS2 topics simulated from Gazebo and attaches covariance to them',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            "covariance_filter_node = covariance_filter.covariance_filter:main"
        ],
    },
)
