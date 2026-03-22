import os, json
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DIR = os.path.join(CURRENT_DIR, '..', 'src', 'drl_policy', 'recorded_paths')

path_list = ['SAC_001_path_1', 'TD3_002_path_1']

def main():
    plt.figure(figsize=(10,10))
    for path in path_list:
        PATH_PATH = os.path.join(PATH_DIR, f'{path}.json')
        MODEL_NAME = path.split('_')[0]

        with open(PATH_PATH, 'r') as f:
            poses = json.load(f)["poses"]

        x   = [pose.get('x', float('nan')) for pose in poses]
        y   = [pose.get('y', float('nan')) for pose in poses]
        yaw = [pose.get('yaw', float('nan')) for pose in poses]

        plt.scatter(x, y, s=5, label=MODEL_NAME)

    plt.legend()
    plt.show()
        

if __name__ == "__main__":
    main()