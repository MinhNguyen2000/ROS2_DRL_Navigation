"""
this wrapper is for adding randomization to the position of the agent, goal, and obstacles. the idea
is that it randomizes the position of these three items based on a given frequency

"""
# import these packages:
import gymnasium as gym

# define the wrapper class:
class RandomizeWrapper(gym.Wrapper):
    # constructor:
    def __init__(self, 
                 env, 
                 agent_freq: int = 1,
                 goal_freq: int = 100,
                 obstacle_freq: int = 250):
        # inherit from env:
        super().__init__(env)

        # define parameters:
        self.env = env
        self.agent_freq = agent_freq
        self.goal_freq = goal_freq
        self.obstacle_freq = obstacle_freq
        self.episode_counter = 0

    # reset function:
    def reset(self, **kwargs):
        # reset conditions:
        agent_randomize = False
        goal_randomize = False
        obstacle_randomize = False

        # advance the episode counter:
        self.episode_counter += 1

        # randomize the agent position based on frequency:
        if int(self.agent_freq) % self.episode_counter == 0:
            # self.randomize_current_goal = True
            agent_randomize = True
        
        # randomize the goal position based on frequency:
        if int(self.goal_freq) % self.episode_counter == 0:
            goal_randomize = True

        # randomize the obstacle position based on frequency:
        if int(self.obstacle_freq) % self.episode_counter == 0:
            obstacle_randomize = True

        # reset as normal:
        obs, info = self.env.reset(agent_randomize = agent_randomize,
                                   goal_randomize = goal_randomize,
                                   obstacle_randomize = obstacle_randomize)
    
        # return:
        return obs, info
