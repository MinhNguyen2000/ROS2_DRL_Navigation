from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'drl_policy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # include the model for DRL inference
        *[
            (
                os.path.join('share', package_name, os.path.dirname(f)),
                [f]
            )
            for f in glob('policy/**/*', recursive=True)
            if os.path.isfile(f)
        ]
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='mnguyen6@unb.ca',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'policy_node = drl_policy.policy_node:main',
            'goal_client = drl_policy.goal_client:main',
            'goal_sequence_server = drl_policy.goal_sequence_server:main',
            'goal_sequence_client = drl_policy.goal_sequence_client:main',
        ],
    },
)
