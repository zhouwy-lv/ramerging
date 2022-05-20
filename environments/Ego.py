import math
import os, sys
from environments.config import Settings
import numpy as np

from environments.Vehicle import Vehicle
from collections import deque
# from environments.IDM import IDM

# *******************************
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


class Ego :
    def __init__(self):
        # surrounding vehicle information
        self.curr_leader = None
        self.orig_leader = None
        self.orig_follower = None
        self.trgt_leader = None
        self.trgt_follower = None
        # navigation information
        self.dis2entrance = None
        self.current_acceleration=traci.vehicle.getAcceleration("ego")
        self.current_speed = traci.vehicle.getSpeed("ego")
        self.ego_length = traci.vehicle.getLength("ego")
        self.acc=deque([self.current_acceleration,self.current_acceleration],maxlen=2)
        self.speedfactor = traci.vehicle.getSpeedFactor("ego")
        # self.speedLimit = self.speedfactor * traci.lane.getMaxSpeed(traci.vehicle.getLaneID("ego"))
        self.speedLimit = 30





    def get_ego_speed_from(self,current_speed, current_acceleration):
        # if current_acceleration == np.nan:
        #     current_acceleration = 0
        new_acceleration = current_acceleration
        if new_acceleration > 4.5:
            new_acceleration = 4.5
        elif new_acceleration < -6.0:
            new_acceleration = -6.0
        new_speed = current_speed + new_acceleration * 0.1
        if new_speed > 30:
            new_speed = 30
        elif new_speed < 0:
            new_speed = 0
        return float(new_speed),new_acceleration

    def get_ego_jerk(self,action):
        # self.current_acceleration = traci.vehicle.getAcceleration("ego")
        # action=action[0]
        # action= np.nan

        # jerk=jerk[0]
        self.current_speed = traci.vehicle.getSpeed("ego")
        new_speed,new_acc = self.get_ego_speed_from(self.current_speed, action)
        self.acc.append(new_acc)
        jerk=(self.acc[-1]-self.acc[-2])/0.1
        return new_speed, float(jerk)

    def get_ego_informataion(self):
        return traci.vehicle.getAccel("ego"),traci.vehicle.getPosition("ego")


    def get_ego_start_speed(self):
        if Settings.RANDOMIZE_START_SPEED:
            start_speed = np.random.normal(Settings.START_SPEED, Settings.START_SPEED_VARIANCE)
            start_speed = np.clip(start_speed, Settings.MIN_START_SPEED, Settings.MAX_START_SPEED)
        else:
            start_speed = Settings.START_SPEED
        return start_speed

    def remove_ego_car():
        if 'ego' in traci.vehicle.getIDList():
            traci.vehicle.remove("ego")
