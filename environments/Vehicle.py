import math
import os, sys
from collections import deque
# *******************************
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    print('success')
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci


class Vehicle:
    def __init__(self, veh_id):
        # general properties
        self.veh_id = veh_id
        self.curr_laneIndex = traci.vehicle.getLaneIndex(self.veh_id)
        self.length = traci.vehicle.getLength(self.veh_id)
        self.width = traci.vehicle.getWidth(self.veh_id)
        # longitudinal properties
        self.pos_longi = traci.vehicle.getLanePosition(self.veh_id)
        self.speed = traci.vehicle.getSpeed(veh_id)
        self.acce = traci.vehicle.getAcceleration(veh_id)
        self.delta_acce = 0
        # self.acce_deque = deque([self.acce, self.acce], maxlen=2)
        # lateral properties
        traci.vehicle.setLaneChangeMode(veh_id, 0)  # 768

    def update_info(self, rd, veh_dict):

        self.pos = traci.vehicle.getPosition(self.veh_id)
        self.speed = traci.vehicle.getSpeed(self.veh_id)
        self.acce = traci.vehicle.getAcceleration(self.veh_id)
        # self.delta_acce = (self.acce_deque[-1] - self.acce_deque[-2]) / 0.1
        # update lateral properties
        # self.pos_lat = traci.vehicle.getLateralLanePosition(self.veh_id) + (self.curr_laneIndex+0.5)*rd.laneWidth
        # self.pos_lat_deque.append(self.pos_lat)
        # self.speed_lat = (self.pos_lat_deque[-1] - self.pos_lat_deque[-2]) / 0.1
        # self.acce_lat = (self.pos_lat_deque[-1] - 2*self.pos_lat_deque[-1] + self.pos_lat_deque[-2]) / 0.01
