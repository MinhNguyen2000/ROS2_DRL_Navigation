"""
this is matt's file for testing mujoco development.

the purpose of this file is for rendering the developed mujoco environment.

"""
# imports:
import mujoco
import mujoco.viewer

# create model:
model  = mujoco.MjModel.from_xml_path("python/environments/assets/env_test_matt.xml")
data = mujoco.MjData(model)

# render:
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

