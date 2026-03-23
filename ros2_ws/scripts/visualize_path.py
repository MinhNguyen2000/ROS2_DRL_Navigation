import os, json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CURRENT_DIR         = os.path.dirname(os.path.abspath(__file__))
SAVED_PATHS_DIR     = os.path.join(CURRENT_DIR, '..', 'src', 'drl_policy', 'recorded_paths')
DEFAULT_PATHS_PATH   = os.path.join(CURRENT_DIR, '..', 'src', 'drl_policy', 'paths', 'paths.json')

path_list = ['SAC_001_path_1', 'TD3_002_path_1']
odom_init = {'x': -3.0, 'y': -3.0}

def main():
    fig, ax = plt.subplots(figsize=(10,10))

    # --- Visualize the path ---
    for path in path_list:
        PATHS_PATH = os.path.join(SAVED_PATHS_DIR, f'{path}.json')
        MODEL_NAME = path.split('_')[0]

        with open(PATHS_PATH, 'r') as f:
            data = json.load(f)
            
        poses = data["poses"]
        x   = [pose.get('x', float('nan')) + odom_init['x'] for pose in poses]
        y   = [pose.get('y', float('nan')) + odom_init['y'] for pose in poses]
        yaw = [pose.get('yaw', float('nan')) for pose in poses]

        elapsed_time = data["elapsed_time"]
        total_distance = data["total_distance"]

        ax.plot(x, y, '--', alpha=0.7, lw=3, label=f"{MODEL_NAME} | {elapsed_time: 6.2f}s | {total_distance: 6.2f}m")

    # --- TODO: Visualize the obstacles ---
    WORLD = 'world_1'
    SDF_PATH = os.path.join(CURRENT_DIR, '..', 'src', 'agent_bringup', 'worlds', f'{WORLD}.sdf')
    obstacles = parse_sdf_obstacles(SDF_PATH)

    for obs in obstacles:
        if obs["type"] == 'box':
            rect = patches.Rectangle(
                xy=(obs['x'] - obs['sx']/2, obs['y'] - obs['sy']/2),
                width=obs['sx'], height=obs['sy'],
                color='red', alpha=0.5, linewidth=2
            )
            ax.add_patch(rect)

        elif obs["type"] == 'cylinder':
            circle = patches.Circle(
                xy = (obs['x'], obs['y']),
                radius = obs['radius'],
                color='red', alpha=0.5, linewidth=2
            )
            ax.add_patch(circle)

    # --- Visualize the goals ---       
    path_name = path_list[0].split('_')[-2:]
    path_name = "_".join(path_name)
    print(f'Comparing the policy performance on {path_name}')
    with open(DEFAULT_PATHS_PATH, 'r') as f:
        AVAILABLE_PATHS = json.load(f)
    goal_x = [goal[0] + odom_init['x'] for goal in AVAILABLE_PATHS[path_name]]
    goal_y = [goal[1] + odom_init['y'] for goal in AVAILABLE_PATHS[path_name]]
    goal_tolerance = 0.1        # meter
    goal_scatter_size = goal_tolerance * 100 / (2.54 / 72)     # one scatter size = 1/72 of an inch
    ax.scatter(goal_x, goal_y, linewidths=1, edgecolors="#1C7826", s=goal_scatter_size, c="#3EA047", alpha=0.5)

    axis_limits = (-4.0, 4.0)
    ax.set_xlim(axis_limits); ax.set_ylim(axis_limits)
    ax.minorticks_on() 
    ax.tick_params(axis='both', which='minor', direction='in')
    ax.tick_params(axis='both', which='major', direction='in')
    ax.grid(False)
    ax.legend()
    plt.show()
        
import xml.etree.ElementTree as ET
def parse_sdf_obstacles(sdf_path: str) -> list:
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    obstacles = []

    for elem in root.iter('model'):
        # print(f"Tag: {elem.tag}, Attributes: {elem.attrib}")   # debug print to show the main tags under <sdf>
        
        name = elem.get('name','')
        is_box = name.startswith('box')
        is_cylinder = name.startswith('cylinder')

        if not (is_box or is_cylinder):
            continue

        pose_elem = elem.find('pose')
        pose_vals = [float(v) for v in pose_elem.text.strip().split()]
        x, y = pose_vals[0], pose_vals[1]

        if is_box:
            size_elem = elem.find('.//collision/geometry/box/size')
            size_x, size_y, _ = [float(v) for v in size_elem.text.strip().split()]
            obstacles.append({
                "name": name,
                "type": "box",
                "x": x,
                "y": y,
                "sx": size_x,
                "sy": size_y
            })
        elif is_cylinder:
            radius_elem = elem.find('.//collision/geometry/cylinder/radius')
            obstacles.append({
                "name": name,
                "type": "cylinder",
                "x": x,
                "y": y,
                "radius": float(radius_elem.text.strip())
            })

    return obstacles


if __name__ == "__main__":
    main()