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

        Args:
            params:     a dict that contains the relevant parameters for creating the environment,
                        such as the: ``env_settings``, ``compiler_settings``, ``option_settings``,
                        ``default_settings``, ``visual_settings``, ``skybox_settings``, ``light_settings``,
                        ``camera_settings``, ``wall_settings``, ``ground_plane_settings``, ``agent_settings``, 
                        and the ``task_settings``.      

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

        # # agent settings:
        # self.agent_name = params["agent_settings"]["name"]
        # self.agent_radius = params["agent_settings"]["radius"]
        # self.agent_height = params["agent_settings"]["height"]
        # self.agent_contype = params["agent_settings"]["contype"]
        # self.agent_conaffinity = params["agent_settings"]["conaffinity"]
        # self.agent_rgba = params["agent_settings"]["rgba"]

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
        self.spec.add_mesh(name = self.mesh_name,
                           file = os.path.join(os.getcwd(), self.mesh_file_name),
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
        this function spawns an agent in the environment. it takes in the position of the agent, and uses the agent specific 
        parameters that are contained within the parameters file to create the agent. this agent is based on the 3rd revision done
        by Minh, and as such does not contain any actuators or nesting of bodies. 

        Args:
            agent_pos:      a list containing the position of the agent, in format ``[X, Y, Z]``
        
        """
        # NEW FORMULATION:
        self.agent = self.spec.worldbody.add_body(name = self.agent_name, pos = agent_pos)
        self.agent.add_joint(name = "agent_x_slide", type = mj.mjtJoint.mjJNT_SLIDE, axis = [1, 0, 0])
        self.agent.add_joint(name = "agent_y_slide", type = mj.mjtJoint.mjJNT_SLIDE, axis = [0, 1, 0])
        self.agent.add_joint(name = "agent_z_yaw", type = mj.mjtJoint.mjJNT_HINGE, axis = [0, 0, 1])

        self.agent.add_geom(name = self.mesh_name,
                            type = mj.mjtGeom.mjGEOM_MESH, 
                            meshname = self.mesh_name,
                            contype = self.mesh_contype,
                            conaffinity = self.mesh_conaffinity,
                            euler = self.mesh_euler,
                            pos = self.mesh_pos
                            )
        
        self.agent_footprint = self.agent.add_body(name = self.footprint_name)
        self.agent_footprint.add_geom(name = self.footprint_name, 
                                    type = mj.mjtGeom.mjGEOM_CYLINDER,
                                    size = [self.footprint_radius, self.footprint_height, 0], 
                                    contype = self.footprint_contype,
                                    conaffinity = self.footprint_conaffinity,
                                    rgba = self.footprint_rgba)

        # self.agent.add_geom(name = "agent_body", 
        #                     type = mj.mjtGeom.mjGEOM_CYLINDER, 
        #                     size = [self.agent_radius, self.agent_height, 0], 
        #                     contype = self.agent_contype, 
        #                     conaffinity = self.agent_conaffinity, 
        #                     rgba = self.agent_rgba)

    # function for adding the lidar:
    def add_lidar(self, n_rays: int):
        """ 
        this function takes in a desired number of rays and uses it to calculate an angular resolution for placing
        that many rays. it does this by looping over the desired number of rays, and placing a site for every angular
        increment. it then binds a rangefinder to this site, before looping over n_rays.

        for instance, n_rays of 10 would mean that there are 36 equally spaced sites + rangefinders around the agent

        Args:
            n_rays:     an int representing the desired number of LiDAR rangefinder rays
        
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
        this function spawns a task in the environment based on provided position.

        Args:
            task_pos:       a list containing the position of the task, in format ``[X, Y, Z]``
        
        """
        task = self.spec.worldbody.add_body(name = "goal", pos = task_pos)
        task.add_geom(name = "goal", type = mj.mjtGeom.mjGEOM_CYLINDER, size = [self.task_radius, self.task_height, 0], contype = 0, conaffinity = 0, rgba = [0, 1, 0, 1])
        task.add_joint(name = "goal_x_slide", type = mj.mjtJoint.mjJNT_SLIDE, axis = [1, 0, 0])
        task.add_joint(name = "goal_y_slide", type = mj.mjtJoint.mjJNT_SLIDE, axis = [0, 1, 0])

    # function for compiling the model:
    def compile(self):
        """ 
        this function compiles the model using the builtin method for ``mj.MjSpec()``, ``.compile()``. the spec must be compiled
        such that it can be used in a broader MuJoCo simulation context.

        """
        self.model = self.spec.compile()

    # not sure if a recompile function is needed, that is the rationale behind splitting up make_spec and compile

    # function for making the environment:
    def make_env(self, agent_pos: list, task_pos: list, n_rays: int):
        """ 
        this function uses the methods defined above and basically just chains them together to make and compile the environment.
        it is responsible for making the ``spec`` and applying the default settings (options, visual, lighting, camera, skybox, plane, walls),
        adding in the agent, adding in the LiDAR, adding the task, and then compiling the ``spec`` into a usable ``model``.

        Args:
            agent_pos:          a list containing the position of the robot, in format ``[X, Y]``
            task_pos:           a list containing the position of the task, in format ``[X, Y]``
            n_rays:             an int specifying the desired number of rays for the LiDAR simulation
        
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

        # compile into model:
        self.compile()

    # function for rendering the environment:
    def render(self):
        """ 
        this function renders and steps through the environment every timestep. it takes the compiled model and extracts
        the data struct, which contains the simulation states. it then launches a viewer using the model and the data.

        The settings that are altered are:
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