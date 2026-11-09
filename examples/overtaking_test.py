import gym
import numpy as np
import yaml
import time
from argparse import Namespace
import tensorflow as tf  # Added to load the paper's model
from collections import deque

from waypoint_follow import PurePursuitPlanner 

def main():
    #main function follow /example/waypoint_follow.py logig

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 1.375}
    
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # 1. LOAD THE PAPER'S PRE-TRAINED MODEL
    #https://github.com/JZ76/f1tenth_simulator_two_agents/tree/master/overtaking_models
    overtaking_model = tf.keras.models.load_model('overtaking_models/model_RNN_10_3.h5')

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=2, timestep=0.01)

    #initial poses are first point of the waypoint.csv file
    pos1 = np.array([conf.sx, conf.sy, conf.stheta ])
    #ego car starts a bit behind
    pos2 = np.array([conf.sx, conf.sy - 5, conf.stheta])
    initial_poses = np.array([
        pos2, # Car 0 (Ego / Behind)
        pos1  # Car 1 (Obstacle / Front)
    ])
    
    #Car 1 is unaware of the othre one, and follows the waypoints.csv using a Pure pursuit
    #(No lidar input)
    planner_car1 = PurePursuitPlanner(conf, (0.17145+0.15875))

    obs, step_reward, done, info = env.reset(initial_poses)
    env.render()

    #from official github repo
    #https://github.com/JZ76/f1tenth_simulator_two_agents/blob/master/node/ML_overtaking_red.py
    size_of_input = 6
    lidar_queue = deque(maxlen=size_of_input)
    state_queue = deque(maxlen=size_of_input)
    
    car0_prev_steer = 0.0
    step_counter = 0

    car0_vel_optimal = 1.0
    car0_steer_optimal = 0.0

    while not done:
        x1, y1, theta1 = obs['poses_x'][1], obs['poses_y'][1], obs['poses_theta'][1]
        ego_speed = obs['linear_vels_x'][0]

        car1_vel_optimal, car1_steer = planner_car1.plan(x1, y1, theta1, work['tlad'], work['vgain'])
        car1_vel_optimal *=0.6    
        # Match ROS 40Hz LiDAR rate inside Gym's 100Hz loop
        if step_counter % 2 == 0:
            
            # 1. MIRROR SENSORS: Flip the LiDAR so Left becomes Right,
            # Pure Pursuit was following the track anti-clock wise, but
            # the NN clock clock wise. si
            ego_lidar = np.clip(obs['scans'][0], 0.0, 10.0) 
            ego_lidar = np.flip(ego_lidar) # <--- FLIP 1
            ego_lidar_padded = np.append(ego_lidar, ego_lidar[-1])
            lidar_queue.append(ego_lidar_padded)
            
            # 2. MIRROR MEMORY: Invert the previous steering angle so the 
            # NN's memory matches the flipped sensors
            kinematics_raw = np.array([ego_speed, -car0_prev_steer]) # <--- FLIP 2
            state_queue.append(kinematics_raw)
            
            if len(lidar_queue) == size_of_input:
                inputA = np.array(lidar_queue).reshape(1, size_of_input, -1)
                inputB = np.array(state_queue).reshape(1, size_of_input, -1)
                
                prediction = overtaking_model([inputA, inputB], training=False).numpy()
                
                #normalizing NN output.
                #paper's default = 17, and 0.24 respectively, but the car crashed in the first turn,
                # so i lowered the speed, and increased steering strentgh 
                norm_vel = 8
                norm_steer = -0.4 #must always be negative!
                car0_vel_optimal = float(prediction[0, -1, 0]) * norm_vel
                
                # 3. MIRROR OUTPUT: Invert the NN's chosen steering command 
                # back to the real world's coordinate system!
                raw_ai_steer = float(prediction[0, -1, 1])
                car0_steer_optimal = raw_ai_steer * norm_steer # <--- FLIP 3
                
            car0_prev_steer = car0_steer_optimal

        step_counter += 1

        action = np.array([
            [car0_steer_optimal, car0_vel_optimal],
            [car1_steer, car1_vel_optimal]
        ])
        
        obs, step_reward, done, info = env.step(action)
        env.render(mode='human')

if __name__ == '__main__':
    main()