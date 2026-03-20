from setuptools import find_packages, setup

package_name = 'drl_gui'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='matthew',
    maintainer_email='mtidd2@unb.ca',
    description='This package launches a GUI that serves as the main modality for interacting with the policy node and deploying the trained   DRL policy on the agent',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gui_node = drl_gui.gui_node:main'
        ],
    },
)
