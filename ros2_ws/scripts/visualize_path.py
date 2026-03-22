import os, json
import matplotlib.pyplot as plt

CURRENT_DIR         = os.path.dirname(os.path.abspath(__file__))
SAVED_PATHS_DIR     = os.path.join(CURRENT_DIR, '..', 'src', 'drl_policy', 'recorded_paths')
DEFAULT_PATHS_PATH   = os.path.join(CURRENT_DIR, '..', 'src', 'drl_policy', 'paths', 'paths.json')

path_list = ['SAC_001_path_1', 'TD3_002_path_1']

def main():
    plt.figure(figsize=(10,10))

    # --- Visualize the path ---
    for path in path_list:
        PATHS_PATH = os.path.join(SAVED_PATHS_DIR, f'{path}.json')
        MODEL_NAME = path.split('_')[0]

        with open(PATHS_PATH, 'r') as f:
            poses = json.load(f)["poses"]

        x   = [pose.get('x', float('nan')) for pose in poses]
        y   = [pose.get('y', float('nan')) for pose in poses]
        yaw = [pose.get('yaw', float('nan')) for pose in poses]

        plt.plot(x, y, '--', lw=2, label=MODEL_NAME)

    # --- TODO: Visualize the obstacles ---

    # --- Visualize the goals ---
    path_name = path_list[0].split('_')[-2:]
    path_name = "_".join(path_name)
    print(f'Comparing the policy performance on {path_name}')
    with open(DEFAULT_PATHS_PATH, 'r') as f:
        AVAILABLE_PATHS = json.load(f)
    goal_x = [goal[0] for goal in AVAILABLE_PATHS[path_name]]
    goal_y = [goal[1] for goal in AVAILABLE_PATHS[path_name]]
    goal_tolerance = 0.1        # meter
    goal_scatter_size = 0.1 * 100 / (2.54 / 72)     # one scatter size = 1/72 of an inch
    plt.scatter(goal_x, goal_y, linewidths=1, edgecolors="#1C7826", s=goal_scatter_size, c="#3EA047", alpha=0.5)

    plt.legend()
    plt.show()
        

if __name__ == "__main__":
    main()