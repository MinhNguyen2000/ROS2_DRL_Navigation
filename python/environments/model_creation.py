"""
this file contains the MakeEnv class.

"""
# imports:
import mujoco as mj
import mujoco.viewer
import numpy as np
import os

# define main class for creating the environment:
class MakeEnv:
    """
    this class is for creating environments using the python API for MuJoCo.
    
    """
    # constructor:
    def __init__(self, 
                 params : dict):
        """ 
        this is the constructor for the class, which does the instantiation of the environment.

        Arguments:
            :param params: a dict that contains the relevant parameters for creating the environment.
            :type params: dict

        """
        # add params to self:
        self.params = params

        ### OBJECT PARAMETERS ###
        # env settings:
        self.env_name = params["env_settings"]["name"]

        # compiler settings:
        self.compiler_angle = params["compiler_settings"]["compiler_angle"]

        # option settings:
        self.timestep = params["option_settings"]["timestep"]
        self.integrator = params["option_settings"]["integrator"]
        self.gravity = params["option_settings"]["gravity"]

        # default settings:
        self.joint_damping = params["default_settings"]["joint_damping"]

        # visual settings:
        self.znear = params["visual_settings"]["znear"]
        self.zfar = params["visual_settings"]["zfar"]
        self.shadowsize = params["visual_settings"]["shadowsize"]
        self.framelength = params["visual_settings"]["framelength"]
        self.framewidth = params["visual_settings"]["framewidth"]
        self.jointlength = params["visual_settings"]["jointlength"]
        self.jointwidth = params["visual_settings"]["jointwidth"]

        # skybox settings:
        self.skybox_name = params["skybox_settings"]["name"]
        self.skybox_type = mj.mjtTexture.mjTEXTURE_SKYBOX
        self.skybox_builtin = mj.mjtBuiltin.mjBUILTIN_GRADIENT
        self.skybox_rgb1 = params["skybox_settings"]["rgb1"]
        self.skybox_rgb2 = params["skybox_settings"]["rgb2"]
        self.skybox_width = params["skybox_settings"]["width"]
        self.skybox_height = params["skybox_settings"]["height"]

        # light settings:
        self.light_name = params["light_settings"]["name"]
        self.light_pos = params["light_settings"]["pos"]
        self.light_diffuse = params["light_settings"]["diffuse"]
        self.light_specular = params["light_settings"]["specular"]
        self.light_ambient = params["light_settings"]["ambient"]

        # camera settings:
        self.camera_name = params["camera_settings"]["name"]
        self.camera_pos = params["camera_settings"]["pos"]

        # wall settings:
        self.wall_type = mj.mjtGeom.mjGEOM_BOX
        self.wall_contype = params["wall_settings"]["contype"]
        self.wall_conaffinity = params["wall_settings"]["conaffinity"]
        self.wall_thickness = params["wall_settings"]["thickness"]
        self.wall_height = params["wall_settings"]["height"]
        self.wall_indices = [("right", [1, 0, 0]),
                             ("left", [-1, 0, 0]),
                             ("front", [0, 1, 0]),
                             ("back", [0, -1, 0])]
        
        # ground plane settings:
        self.ground_name = params["ground_settings"]["name"]
        self.ground_type = mj.mjtGeom.mjGEOM_PLANE
        self.ground_contype = params["ground_settings"]["contype"]
        self.ground_conaffinity = params["ground_settings"]["conaffinity"]
        self.ground_internal_length = params["ground_settings"]["internal_length"]
        self.ground_actual_length = self.ground_internal_length + 2 * self.wall_thickness
        self.ground_z_spacing = params["ground_settings"]["z_spacing"]
        self.ground_size = [self.ground_actual_length, self.ground_actual_length, self.ground_z_spacing]
        self.ground_pos = params["ground_settings"]["pos"]
        self.ground_rgba = params["ground_settings"]["rgba"]
        
        # global agent settings:
        self.agent_name = params["agent_settings"]["name"]

        # agent mesh settings:
        self.mesh_name = params["agent_mesh_settings"]["name"]
        self.mesh_file_name = params["agent_mesh_settings"]["file_name"]
        self.mesh_contype = params["agent_mesh_settings"]["contype"]
        self.mesh_conaffinity = params["agent_mesh_settings"]["conaffinity"]
        self.mesh_euler = params["agent_mesh_settings"]["euler"]
        self.mesh_pos = params["agent_mesh_settings"]["pos"]

        # agent footprint settings:
        self.footprint_name = params["agent_footprint_settings"]["name"]
        self.footprint_radius = params["agent_footprint_settings"]["radius"]
        self.footprint_height = params["agent_footprint_settings"]["height"]
        self.footprint_contype = params["agent_footprint_settings"]["contype"]
        self.footprint_conaffinity = params["agent_footprint_settings"]["conaffinity"]
        self.footprint_rgba = params["agent_footprint_settings"]["rgba"]

        # obstacle settings:
        self.obstacle_counter = 0
        self.obstacle_footprint_size = params["obstacle_settings"]["footprint_size"]
        self.obstacle_thickness = params["obstacle_settings"]["thickness"]
        self.obstacle_height = params["obstacle_settings"]["height"]
        self.obstacle_size_low = params["obstacle_settings"]["size_low"]
        self.obstacle_size_high = params["obstacle_settings"]["size_high"]

        # task settings:
        self.task_radius = params["task_settings"]["radius"]
        self.task_height = params["task_settings"]["height"]

    # function for initializing the MjSpec:
    def make_spec(self):
        """ 
        this function initializes the ``MjSpec`` and applies the passed basic settings/requirements for the 
        environment (plane, skybox, light, camera, walls, etc.).

        """
        # initialize spec:
        self.spec = mj.MjSpec()

        # set the compiler settings:
        self.spec.compiler.degree = self.compiler_angle

        # set the option settings:
        self.spec.option.timestep = self.timestep
        self.spec.option.integrator = self.integrator
        self.spec.option.gravity = self.gravity
        
        # set the visualization settings:
        self.spec.visual.quality.shadowsize = self.shadowsize
        self.spec.visual.map.znear = self.znear
        self.spec.visual.map.zfar = self.zfar
        self.spec.visual.scale.framelength = self.framelength
        self.spec.visual.scale.framewidth = self.framewidth
        self.spec.visual.scale.jointlength = self.jointlength
        self.spec.visual.scale.jointwidth = self.jointwidth

        # set the default settings:
        self.spec.default.joint.damping = self.joint_damping

        # add the skybox:
        self.spec.add_texture(name = self.skybox_name,
                              type = self.skybox_type,
                              builtin = self.skybox_builtin,
                              width = self.skybox_width, 
                              height = self.skybox_height, 
                              rgb1 = self.skybox_rgb1,
                              rgb2 = self.skybox_rgb2)
        
        # add the mesh:
        base_path = os.path.dirname(os.path.abspath(__file__))
        mesh_path = os.path.join(base_path, self.mesh_file_name)
        self.spec.add_mesh(name = self.mesh_name,
                           file = mesh_path,
                           scale = [1/1000, 1/1000, 1/1000])
        
        # add the light:
        self.spec.worldbody.add_light(name = self.light_name,
                                 pos = self.light_pos, 
                                 diffuse = self.light_diffuse,
                                 specular = self.light_specular, 
                                 ambient = self.light_ambient)
        
        # add camera:
        self.spec.worldbody.add_camera(name = self.camera_name,
                                       pos = self.camera_pos)
        
        # add ground plane:
        self.spec.worldbody.add_geom(name = self.ground_name,
                                     type = self.ground_type,
                                     contype = self.ground_contype,
                                     conaffinity = self.ground_conaffinity,
                                     pos = self.ground_pos,
                                     size = self.ground_size, 
                                     rgba = self.ground_rgba)
        
        # add walls:
        for name, axis in self.wall_indices:
            # if its an x-wall:
            if abs(axis[0]):
                wall_size = [self.wall_thickness, self.ground_actual_length, self.wall_height]
                wall_pos = [axis[0] * (self.ground_actual_length - self.wall_thickness), 0, self.wall_height]
            # else its a y-wall:
            else:
                wall_size = [self.ground_actual_length - 2 * self.wall_thickness, self.wall_thickness, self.wall_height]
                wall_pos = [0, axis[1] * (self.ground_actual_length - self.wall_thickness), self.wall_height]

            # add the geom to the spec
            self.spec.worldbody.add_geom(name = name,
                                        type = self.wall_type,
                                        contype = self.wall_contype,
                                        conaffinity = self.wall_conaffinity,
                                        pos = wall_pos,
                                        size = wall_size)

    # function for adding in an agent:
    def add_agent(self, agent_pos: list):
        """ 
        this function spawns an agent in the environment, taking in the position of the agent, and uses the agent specific parameters
        from the params dict to create the agent.

        Arguments:
            :param agent_pos: a list containing the position of the agent, in format ``[X, Y, Z]``.
            :type agent_pos: list

        """
        # add agent to the worldbody:
        self.agent = self.spec.worldbody.add_body(name = self.agent_name, pos = agent_pos)

        # add joints to the agent such that it can translate/rotate:
        self.agent.add_joint(name = "agent_x_slide", type = mj.mjtJoint.mjJNT_SLIDE, axis = [1, 0, 0])
        self.agent.add_joint(name = "agent_y_slide", type = mj.mjtJoint.mjJNT_SLIDE, axis = [0, 1, 0])
        self.agent.add_joint(name = "agent_z_yaw", type = mj.mjtJoint.mjJNT_HINGE, axis = [0, 0, 1])

        # add the geom to the agent, so that it may be visualized:
        self.agent.add_geom(name = self.mesh_name,
                            type = mj.mjtGeom.mjGEOM_MESH, 
                            meshname = self.mesh_name,
                            contype = self.mesh_contype,
                            conaffinity = self.mesh_conaffinity,
                            euler = self.mesh_euler,
                            pos = self.mesh_pos
                            )
        
        # add a footprint to the agent:
        self.agent_footprint = self.agent.add_body(name = self.footprint_name)
        self.agent_footprint.add_geom(name = self.footprint_name, 
                                    type = mj.mjtGeom.mjGEOM_CYLINDER,
                                    size = [self.footprint_radius, self.footprint_height, 0], 
                                    contype = self.footprint_contype,
                                    conaffinity = self.footprint_conaffinity,
                                    rgba = self.footprint_rgba)

    # function for adding the lidar:
    def add_lidar(self, n_rays: int):
        """ 
        this function takes in a desired number of rays and uses it to place sensor sites, incremented by an angular resolution that will
        achieve the desired number of rays. it loops over the number of sites and binds a rangefinder to each one. 

        for instance, if ``n_rays`` is 10, there would be 36 equally spaced sites + rangefinders around the agent.

        Arguments:
            :param n_rays: an int representing the desired number of LiDAR rangefinder rays
            :type agent_pos: int

        """
        # add resolution to class:
        self.n_rays = n_rays

        # get number of sites:
        self.resolution = float(360 / self.n_rays)

        # add sites and rangefinders:
        for i in range(self.n_rays + 1):
            # place a site:
            self.agent.add_site(name = f"lidar_site_{i}", pos = [0, 0, 0.0614 + 0.08285], euler = [-90, 90 + i * self.resolution, 0])
            self.spec.add_sensor(name = f"lidar_ray_{i}",
                                 type = mj.mjtSensor.mjSENS_RANGEFINDER, 
                                 objtype = mj.mjtObj.mjOBJ_SITE,
                                 objname = f"lidar_site_{i}")

    # function for adding in a task:
    def add_task(self, task_pos: list):
        """
        this function spawns a goal in the environment based on provided position.

        Arguments:
            :param task_pos: a list containing the position of the task, in format ``[X, Y, Z]``.
            :type task_pos:  list

        """
        # add the task to the worldbody:
        task = self.spec.worldbody.add_body(name = "goal", pos = task_pos)

        # add a geom to the task:
        task.add_geom(name = "goal", type = mj.mjtGeom.mjGEOM_CYLINDER, size = [self.task_radius, self.task_height, 0.0], contype = 0, conaffinity = 0, rgba = [0, 1, 0, 1])
        
        # add joints to the task:
        task.add_joint(name = "goal_x_slide", type = mj.mjtJoint.mjJNT_SLIDE, axis = [1, 0, 0])
        task.add_joint(name = "goal_y_slide", type = mj.mjtJoint.mjJNT_SLIDE, axis = [0, 1, 0])

    # # function for generating obstacle points for composite obstacles:
    # def generate_segment_positions(self, segment_length_range: list):
    #     """
    #     this function generates various wall types within the environment. the shapes that it generates include
    #     straight walls, L-shaped walls, T-shaped walls, and "plus"-shaped walls.

    #     :param segment_length_range: a list containing the range of desired wall lengths
    #     :type segment_length_range: list
    
    #     """
    #     # define a list of obstacle types:
    #     obstacle_types = ["corner", "tee", "plus"]

    #     # sample a random obstacle type:
    #     obstacle_type = np.random.choice(obstacle_types)
    #     # print(f"obstacle is a : {obstacle_type}")

    #     # define an empty list for holding the segments of the obstacle:
    #     segments = []

    #     # while the obstacle has not yet been generated, attempt to generate the obstacle:
    #     # define the center, or origin, of the obstacle:
    #     center = np.array([0, 0])

    #     # sample an angle for the main direction of the obstacle:
    #     initial_angle = 0.0

    #     # define a list for holding the coordinate points of the obstacle, initialized with the center:
    #     points = [center.copy()]

    #     # define an empty list for holding the angles:
    #     angles = []

    #     # match cases for each obstacle type:
    #     match obstacle_type:
    #         # straight wall case:
    #         case "straight":
    #             # sample two lengths:
    #             length = np.random.uniform(*segment_length_range, size = 2)

    #             # need to move one length in each direction from the center point:
    #             for direction, length in zip([-1, 1], length):
    #                 # compute the next point:
    #                 p_new = center + length * direction * np.array([np.cos(initial_angle), np.sin(initial_angle)])

    #                 # append:
    #                 points.append(p_new.copy())
    #                 angles.append(initial_angle)
            
    #         # 90 degree corner case:
    #         case "corner":
    #             # sample two lengths:
    #             length = np.random.uniform(*segment_length_range, size = 2)

    #             # for every angle and length combination:
    #             for i, l in enumerate(length):
    #                 # compute the angle:
    #                 angle = initial_angle + i * (np.pi/2)

    #                 # compute the next point:
    #                 p_new = center + l *np.array([np.cos(angle), np.sin(angle)])

    #                 # append:
    #                 points.append(p_new.copy())
    #                 angles.append(angle)
                    
    #         # tee shaped case:
    #         case "tee":
    #             # sample three lengths:
    #             length = np.random.uniform(*segment_length_range, size = 3)

    #             # for every angle and length combination:
    #             for i, l in enumerate(length):
    #                 # compute the new angle:
    #                 angle = initial_angle + i * (np.pi/2)

    #                 # compute the next point:
    #                 p_new = center + l * np.array([np.cos(angle), np.sin(angle)])

    #                 # append:
    #                 points.append(p_new.copy())
    #                 angles.append(angle)

    #         # plus shaped case:
    #         case "plus": 
    #             # sample four lengths:
    #             length = np.random.uniform(*segment_length_range, size = 4)

    #             # for every angle and length combination:
    #             for i, l in enumerate(length):
    #                 # compute the angle:
    #                 angle = initial_angle + i * (np.pi/2)

    #                 # compute the next point:
    #                 p_new = center + l * np.array([np.cos(angle), np.sin(angle)])

    #                 # append:
    #                 points.append(p_new.copy())
    #                 angles.append(angle)

    #     # order them as segment pairs:
    #     for i in range(len(points) - 1):
    #         segments.append([points[0], points[i+1], angles[i]])

    #     # return to user:
    #     return segments, points, length

    # function for adding in primitive obstacles:
    def add_primitive_obstacle(self, obs_pos: list):
        """
        this function generates a simple, primitive obstacle, either a box or a cylinder.

        Arguments:
            :param obs_pos: a list containing the ``[X, Y, Z]`` position of the obstacle.
            :type obs_pos: list

        """
        # increment the obstacle counter:
        self.obstacle_counter += 1

        # add the obstacle to the worldbody: 
        self.obstacle = self.spec.worldbody.add_body(name = f"obstacle_{self.obstacle_counter}", pos = obs_pos)

        # randomly select a primitive shape:
        geom_type = np.random.choice([mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_BOX])

        # random pose of obstacle:
        obstacle_angle = [0, 0, np.random.choice([-90, 0, 90])]

        ################################################# MAIN BODY #################################################
        # match case for type because sizes differ:
        match geom_type:
            case mj.mjtGeom.mjGEOM_CYLINDER:
                obstacle_size = np.array([np.random.uniform(low = self.obstacle_size_low, high = self.obstacle_size_high), self.obstacle_height, 0.0])
                footprint_size = obstacle_size.copy()
                footprint_size[0] += self.obstacle_thickness * 2
                footprint_size[1] = self.footprint_height

            case mj.mjtGeom.mjGEOM_BOX:
                box_side_length = np.random.uniform(low = self.obstacle_size_low, high = self.obstacle_size_high)
                obstacle_size = np.array([box_side_length, box_side_length, self.obstacle_height])
                footprint_size = obstacle_size.copy()
                footprint_size[0:2] += self.obstacle_thickness * 2
                footprint_size[2] = self.footprint_height

        # add joints to the obstacle:
        self.obstacle.add_joint(name = f"obstacle_{self.obstacle_counter}_x_slide", type = mj.mjtJoint.mjJNT_SLIDE, axis = [1, 0, 0])
        self.obstacle.add_joint(name = f"obstacle_{self.obstacle_counter}_y_slide", type = mj.mjtJoint.mjJNT_SLIDE, axis = [0, 1, 0])

        # add that shape to the environment:
        self.obstacle.add_geom(name = f"obstacle_{self.obstacle_counter}_geom",
                               type = geom_type,
                               size = obstacle_size,
                               euler = obstacle_angle,
                               contype = 1,
                               conaffinity = 1,
                               rgba = [0, 0, 1, 1])
        
        ################################################# FOOTPRINT #################################################
        # add footprint to obstacle body:
        footprint = self.obstacle.add_body(name = f"footprint_{self.obstacle_counter}", pos = [0, 0, self.footprint_height])

        # add dilated geom:
        footprint.add_geom(name = f"obstacle_{self.obstacle_counter}_footprint_geom",
            type = geom_type,
            size = footprint_size,
            pos = [0, 0, -self.obstacle_height + self.footprint_height],
            euler = obstacle_angle,
            contype = 1, 
            conaffinity = 1,
            rgba = [1, 0, 0, 0.1]
        )

    # # function for adding in composite obstacles:
    # def add_composite_obstacle(self, obs_pos: list):
    #     """
    #     this function spawns an obstacle in the environment based on a provided position

    #     Args:
    #         obs_pos:        a list containing the position of the obstacle, in format ``[X, Y, Z]``

    #     """           
    #     # increment obstacle counter:
    #     self.obstacle_counter += 1
        
    #     # add the obstacle to the worldbody:
    #     self.obstacle = self.spec.worldbody.add_body(name = f"obstacle_{self.obstacle_counter}", pos = [obs_pos[0], obs_pos[1], 0])
        
    #     # get the segment pairs of the composite obstacle:
    #     segments, _, length = self.generate_segment_positions(segment_length_range = [0.10, 0.35])

    #     ################################################# MAIN BODY #################################################
    #     # need to place a geom at the center of the segments, spanning the distance covered by the segment:
    #     for i, (p0, p1, angle) in enumerate(segments):
    #         # get difference in x and y:
    #         dx = p1[0] - p0[0]
    #         dy = p1[1] - p0[1]

    #         # get length of segment:
    #         length = np.hypot(dx, dy)

    #         # find midpoints:
    #         mid_x = 0.5 * (p0[0] + p1[0])
    #         mid_y = 0.5 * (p0[1] + p1[1])

    #         # add the box:
    #         self.obstacle.add_geom(
    #             name = f"obstacle_{self.obstacle_counter}_geom_{i + 1}",
    #             type = mj.mjtGeom.mjGEOM_BOX,
    #             pos = [mid_x, mid_y, self.obstacle_height],
    #             size = [length / 2, self.obstacle_thickness/2, self.obstacle_height],
    #             euler = [0, 0, np.rad2deg(angle)],
    #             contype = 1,
    #             conaffinity = 1,
    #             rgba = [0, 0, 1, 1]
    #         )

    #     # add center piece because I am lazy:
    #     self.obstacle.add_geom(
    #         name = f"obstacle_{self.obstacle_counter}_center_geom",
    #         type = mj.mjtGeom.mjGEOM_BOX,
    #         pos = [-self.obstacle_thickness/4, -self.obstacle_thickness/4, self.obstacle_height],
    #         size = [self.obstacle_thickness/4, self.obstacle_thickness/4, self.obstacle_height],
    #         euler = [0, 0, np.rad2deg(angle)],
    #         contype = 1, 
    #         conaffinity = 1,
    #         rgba = [0, 0, 1, 1]
    #     )

    #     ################################################# FOOTPRINT #################################################
    #     footprint = self.obstacle.add_body(name = f"footprint_{self.obstacle_counter}", pos = [0, 0, 0])
            
    #     # dilation:
    #     for i, (p0, p1, angle) in enumerate(segments):
    #         # get difference in x and y:
    #         dx = p1[0] - p0[0]
    #         dy = p1[1] - p0[1]

    #         # get length of segment:
    #         length = np.hypot(dx, dy) + self.obstacle_thickness

    #         # find midpoints:
    #         mid_x = 0.5 * (p0[0] + p1[0])
    #         mid_y = 0.5 * (p0[1] + p1[1])

    #         # add the box:
    #         footprint.add_geom(
    #             name = f"obstacle_{self.obstacle_counter}_footprint_geom_{i + 1}",
    #             type = mj.mjtGeom.mjGEOM_BOX,
    #             pos = [mid_x, mid_y, self.footprint_height],
    #             size = [length / 2, self.obstacle_thickness, self.footprint_height],
    #             euler = [0, 0, np.rad2deg(angle)],
    #             contype = 1,
    #             conaffinity = 1,
    #             rgba = [1, 0, 0, 0.1]
    #         )

    #     # add a center footprint because I am lazy:
    #     footprint.add_geom(
    #         name = f"obstacle_{self.obstacle_counter}_footprint_center_geom",
    #         type = mj.mjtGeom.mjGEOM_BOX,
    #         pos = [0, 0, 0],
    #         size = [self.obstacle_thickness, self.obstacle_thickness, self.footprint_height],
    #         euler = [0, 0, np.rad2deg(angle)],
    #         contype = 1, 
    #         conaffinity = 1,
    #         rgba = [1, 0, 0, 0.1]
    #     )

    # function for compiling the model:
    def compile(self):
        """ 
        this function compiles the model using the builtin method for ``mj.MjSpec()``, ``.compile()``. the spec must be compiled
        such that it can be used in a broader MuJoCo simulation context.

        """
        # compile the spec into a model:
        self.model = self.spec.compile()

    # not sure if a recompile function is needed, that is the rationale behind splitting up make_spec and compile

    # function for making the environment:
    def make_env(self, 
                 agent_pos: list, 
                 task_pos: list, 
                 n_rays: int, 
                 obs_pos : list | None = None):
        """ 
        this function uses the methods defined above and basically just chains them together to make and compile the environment.
        it is responsible for making the ``spec`` and applying the default settings (options, visual, lighting, camera, skybox, plane, walls),
        adding in the agent, LiDAR, obstacles, and task, and then compiling the ``spec`` into a usable ``model``.

        Arguments:
            :param agent_pos:   a list containing the position of the agent, in format ``[X, Y]``.
            :param task_pos:    a list containing the position of the task, in format ``[X, Y]``.
            :param n_rays:      an int specifying the desired number of rays for the simulated LiDAR.
            :param obs_pos:     a list containing the position of the obstacles, in format ``[[X1, Y1], [X2, Y2], ...]``
            
            :type agent_pos: list
            :type task_pos: list
            :type n_rays: int
            :type obs_pos: list
        """
        # verify that the provided agent position is feasible:
        if abs(agent_pos[0]) + self.footprint_radius > self.ground_internal_length or abs(agent_pos[1]) + self.footprint_radius > self.ground_internal_length:
            raise ValueError("Provided position of the agent is outside the internal area of the arena!")
        
        # verify that the provided task position is feasible:
        if abs(task_pos[0]) + self.footprint_radius > self.ground_internal_length or abs(task_pos[1]) + self.footprint_radius > self.ground_internal_length:
            raise ValueError("Provided position of the task is unreachable by the agent!")
        
        # initialize the spec:
        self.make_spec()

        # add the agent:
        self.add_agent(agent_pos = [agent_pos[0], agent_pos[1], self.footprint_height])

        # add the lidar:
        self.add_lidar(n_rays)

        # add the task:
        self.add_task(task_pos = [task_pos[0], task_pos[1], self.task_height])

        # add obstacles:
        n_obstacles = len(obs_pos)

        # for every obstacle:
        for i in range(n_obstacles):
            # add a primitive obstacle:
            self.add_primitive_obstacle(obs_pos = [obs_pos[i][0], obs_pos[i][1], self.obstacle_height])

        # compile into model:
        self.compile()

    # function for rendering the environment:
    def render(self):
        """ 
        this function renders and steps through the environment every timestep. it takes the compiled model and extracts
        the data struct, which contains the simulation states. it then launches a viewer using the model and the data.

        the settings that are altered are:
            viewer.cam.type:                this specifies the type of camera that is used
            viewer.cam.fixedcamid:          this specifies the ID of the user-defined camera
            viewer.opt.frame:               this specifies which frame(s) to have active
            viewer.opt.flags:               this specifies the flag(s) to enable
        
        """
        # get model data:
        self.data = mj.MjData(self.model)

        # launch a passive window using the model and the data contained within:
        with mujoco.viewer.launch_passive(self.model, self.data) as self.viewer:
            # switch the camera:
            self.viewer.cam.type = mj.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = self.model.camera(self.camera_name).id

            # enable viewer options:
            self.viewer.opt.frame = mj.mjtFrame.mjFRAME_BODY
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
            
            # while viewer is active, step the model every timestep:
            while self.viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()