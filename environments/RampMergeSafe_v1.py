import os, sys, random, datetime, gym, math
from time import time, sleep

from gym import spaces
from gym.utils import seeding

import numpy as np
from environments.Road import Road
from environments.Vehicle import Vehicle
from environments.Ego import Ego
from environments import control

import traci

class RampMergingEnv(gym.Env):
    def __init__(self, gui=False, max_timesteps=500, label='default', is_train=True):
        self.max_timesteps = max_timesteps
        self.is_train = is_train
        self.min_acce = -4.5
        self.max_acce = 2.6
        self.egoID='ego'
        self._max_episode_steps =500
        self.action_space = spaces.Box(low=self.min_acce, high=self.max_acce, shape=(1,))
        # self.action_space = spaces.Discrete(6)
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(21,))
        self.state=None
        self.sumoBinary = os.environ["SUMO_HOME"] + '/bin/sumo'
        self.sumoCmd_base = ['--lateral-resolution', str(0.8),  # using 'Sublane-Model'
                             '--step-length', str(0.1),
                             '--default.action-step-length', str(0.1),
                             '--no-warnings', str(True),
                             '--no-step-log', str(True)]
        if gui:
            self.sumoBinary += '-gui'
            self.sumoCmd_base += ['--quit-on-end', str(True), '--start', str(True)]
        sumoCmd = [self.sumoBinary] + ['-c', '../map/ramp3/mapDense.sumo.cfg'] + self.sumoCmd_base
        traci.start(sumoCmd, label=label)
        self.rd = Road()
        self.seed()

    # def seed(self, seed=None):
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def distance (self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def step(self, action):

        # start_velocity=control.get_ego_start_speed()
        # control.add_ego_car(start_velocity)
        # if action not in self.action_space[0]:
        #     action= 0
        # traci.vehicle.setDecel()
        # traci.vehicle.setAccel("ego",action)
        self.veh_allList = traci.vehicle.getIDList()


        assert self.egoID in self.veh_allList, 'vehicle not in env'
        self.merge_pos = self.rd.mergingPoint
        self.dest_pos = self.rd.destJunction
        self.rampstart = self.rd.rampstartj
        old_speed = traci.vehicle.getSpeed("ego")
        new_acceleration = action
        if new_acceleration > 4.5:
            new_acceleration = 4.5
        elif new_acceleration < -6.0:
            new_acceleration = -6.0
        current_speed= old_speed + new_acceleration * 0.1

        ego_Point = traci.vehicle.getPosition(self.egoID)

        self.dis2dest = self.distance(ego_Point,self.dest_pos)
        self.dist2merge = self.distance(ego_Point,self.merge_pos)
        self.cos = (abs(ego_Point[0] - self.merge_pos[0]))/self.dist2merge

        # print('theta_ramp:{}  cos:{}  selfcos:{}'.format(theta_ramp,math.cos(theta_ramp),self.cos))
        self.FalsevehList= self.getclosecar(self.veh_allList,ego_Point,current_speed,action_constraint=True)
        _,_,front_distance,back_distance,front_speed,back_speed=self.get_TTC(action_constraint=True)

        if ego_Point[0] - self.merge_pos[0] <0:
            if 0 < front_distance < 10 :
                # if front_distance >5 and front_speed >= 0 and action > 0:
                if front_distance > 5:
                    self.front_danger_action = True
                if front_distance <= 5:
                     action = -4.5
                     self.front_close_danger_action = True
                # self.num_dangeraction += 1

            if 0 < back_distance < 5 and front_distance >= 12:
                # if back_speed >= 0 and  action < 0:
                self.back_danger_action = True
            # if 0 < back_distance < 10 and self.front_close_danger_action == False :
            #      if  back_speed >= 0 and  action < 0:
            #         self.back_danger_action = True
                 # if ego_Point[0] - self.merge_pos[0] < 50:
                 # if back_distance <= 5 :
                 #    action = 2.6
                 #    self.back_close_danger_action = True
                # self.num_dangeraction += 1


        if self.front_close_danger_action or self.front_danger_action :
            self.num_dangeraction += 1
        if self.back_danger_action or self.back_close_danger_action :
            self.num_dangeraction += 1


        new_speed,jerk=Ego().get_ego_jerk(action)
        traci.vehicle.setSpeed("ego", new_speed)
        # print(dis2dest)
        if self.dis2dest < 30:
            print(" **success**" ,self.dis2dest)
            self.is_final_success = True
        # arriveIdlist=traci.simulation.getArrivedIDList()
        if ego_Point[0]- self.merge_pos[0] < 0:
            self.is_success = True


        traci.simulationStep()
        self.veh_allList = traci.vehicle.getIDList()
        # acc= traci.vehicle.getAcceleration(self.egoID)
        ego_pos = traci.vehicle.getPosition(self.egoID)

        colli_num=traci.simulation.getCollidingVehiclesNumber()
        if colli_num>0:
            print('\n collision numbers:',colli_num)
            self.crash = True
            next_state = np.zeros((18,))
            if ego_pos[0]>1000 or ego_pos[0] < -1000:
                # print('--data error--',ego_pos)
                # ego_pos = (923.28,15.11)
                # self.data_error = True
                ego_pos = ego_Point
        else:
            if ego_pos[0]>1000 or ego_pos[0] < -1000:
                # print('--data error--',ego_pos)
                # ego_pos = (923.28,15.11)
                # self.data_error = True
                ego_pos = ego_Point
            next_state = self.update_state(new_speed,jerk,ego_pos)
        # print(next_state)
        rewards = self.update_rw(jerk,new_speed, ego_pos, self.merge_pos, self.dest_pos)

        is_done=self.is_done()

        for i in next_state:
            if np.isnan(i):
                print('state error nan')
            if np.isinf(i):
                print('state error inf')

        for i in next_state:
            if i>=50:
                print('state error big num:{} '.format(i))
                print(next_state)
            if i<=-50:
                print('state error min:{} state'.format(i))
                print(next_state)

        if self.is_danger or self.crash:
            self.info['is_danger']=1
        else:
            self.info['is_danger'] = 0
        # if self.crash:
        #     self.info['crash'] = 1
        # else:
        #     self.info['crash'] = 0
        self.info['crash'] = 0
        if self.is_final_success:
            self.info['is_final_sccess'] = 1
        else:
            self.info['is_final_sccess'] = 0
        if self.is_success:
            self.info['is_success'] = 1
        else:
            self.info['is_success'] = 0
        if self.timestep > self.max_timesteps:
            self.info['time_out'] = 1
        else:
            self.info['time_out'] = 0
        # if self.data_error:
        #     self.info['discard_data'] = 1
        # else:
        #     self.info['discard_data'] = 0

        self.info['numdanger_action'] = self.num_dangeraction


        self.timestep += 1

        return next_state, rewards, is_done, self.info


    def reset(self, tfc=None, is_gui=False, sumoseed=None, randomseed=None,dense=True):
        # tfc = np.random.choice([0, 1, 2]) if tfc is None else tfc
        tfc = 1
        if dense == False:
            if tfc == 0:
                cfg = '../map/ramp3/mapDenseSlow.sumo.cfg'
            elif tfc == 1:
                cfg = '../map/ramp3/mapDense.sumo.cfg'
            else:
                assert tfc == 2
                cfg = '../map/ramp3/mapDenseFast.sumo.cfg'
        else:
            if tfc == 0:
                cfg = '../map/rampdense/mapDenseSlow.sumo.cfg'
            elif tfc == 1:
                cfg = '../map/rampdense/mapDense.sumo.cfg'
            else:
                assert tfc == 2
                cfg = '../map/rampdense/mapDenseFast.sumo.cfg'
        sumoCmd_load = ['-c', cfg] + self.sumoCmd_base
        self.seed(randomseed)
        if not sumoseed:
            sumoCmd_load += ['--random']  # initialize with current system time
        else:
            sumoCmd_load += ['--seed', str(sumoseed)]  # initialize with given seed
        traci.load(sumoCmd_load)

        self.timestep = 0
        self.veh_allList = []
        self.info = {}
        self.crash = False
        self.is_danger = False
        self.is_success = False
        self.is_final_success = False
        self.danger_action = False
        self.num_dangeraction = 0
        self.successflag = 1
        self.front_danger_action = False
        self.front_close_danger_action = False
        self.back_danger_action = False
        self.back_close_danger_action = False
        # self.data_error = False
        # self.done=False
        # traci.simulationStep()
        for i in range(int(6 / 0.1)):
            self.preStep()
        start_velocity = control.get_ego_start_speed()
        control.add_ego_car(start_velocity)
        while self.egoID not in self.veh_allList:
            self.preStep()
            for id in traci.edge.getLastStepVehicleIDs(self.rd.highwayEntEdge):
                traci.vehicle.setLaneChangeMode(id, 0)
            if self.timestep > 500:
                raise Exception('cannot find ego after 1000 timesteps')
        assert self.egoID in self.veh_allList, "cannot start training while ego is not in env"
        # sleep(1)
        # delay()
        a = np.zeros((18,))
        return a


    def close(self):
        traci.close()

    def update_state(self,ego_speed,jerk,ego_Point):
        # self.before_car_point = []
        # self.after_car_point = []
        # ego_acc,ego_Point = Ego().get_ego_informataion()
        # ego_list = [ego_speed/30, jerk/105, (ego_Point[0]-self.rd.mergingPoint[0])/500, (ego_Point[0]-self.rd.destJunction[0])/500]
        # self.ego_x=EgoPoint[0]
        ego_list = [ego_speed/40 , jerk/71 , (ego_Point[0] - self.dest_pos[0])/500,(ego_Point[1]-self.dest_pos[1])/125,
                    (ego_Point[0]-self.merge_pos[0])/500,(ego_Point[1] - self.merge_pos[1])/125]
        # ego_list = [ego_speed / 30, jerk / 105]
        # self.veh_allList= traci.vehicle.getIDList()
        self.vehicle_tuple= self.getclosecar(self.veh_allList,ego_Point,ego_speed)
        # self.before_car_point_diff = self.vehicle_tuple[1]
        # self.after_car_point_diff = -self.vehicle_tuple[5]
        # self.before_speed_diff = self.vehicle_tuple[0]
        # self.after_speed_diff = -self.vehicle_tuple[4]
        next_state=ego_list+self.vehicle_tuple
        return np.array(next_state)


    def update_rw(self,jerk,new_speed,ego_pos,merge_pos,dest_post):
        w_speed = 0.1
        w_safe = 0.1
        w_comf = 0
        w_dest = 0.02
        w_safec = 0.1
        a= 1.5
        # r_speed = -1 + np.exp(-abs(Ego().current_speed - Ego().speedLimit))
        # r_speed = -0.1*abs(Ego().current_speed - Ego().speedLimit)
        # print(self.timestep)

        if self.is_success and self.successflag == 1:
            r_dest = 50
            self.successflag -= 1
            print('\n merge success')
        # elif self.is_success == False :
        #     r_dest = -0.01 * self.distance(ego_pos,dest_post)-0.01*self.distance(ego_pos,merge_pos)
        elif self.is_final_success:
            r_dest = 100
        elif self.timestep > self.max_timesteps:
            r_dest = -100
        else:
            r_dest = 0


        if self.crash or self.is_danger:
            r_safe = -100
        else :
            TTC_front,TTC_back,_,_,_,_ = self.get_TTC(action_constraint=False)
            front_ratio = min(TTC_front / 3,1)
            back_ratio = min(TTC_back / 3, 1)
            # r_safe =-1+np.exp(np.log(front_ratio) + np.log(back_ratio))
            r_safe = np.log(front_ratio) + np.log(back_ratio)
            # r_safe = 0


        front_t2merge = self.t2merge_front[0][0]
        back_t2merge = self.t2merge_back[0][0]
        ego_t2merge = self.tego2merge
        if ego_pos[0]-merge_pos[0] >= 0:
        # if not self.is_success:
            # print(self.is_success)
            if front_t2merge >0:

                tfront_diff = min(abs(ego_t2merge-front_t2merge),a)
            else:
                tfront_diff = a
            if back_t2merge > 0:
                tback_diff = min(abs(back_t2merge-ego_t2merge),a)
            else:
                tback_diff = a
            r_safec = np.log(tfront_diff/a) + np.log(tback_diff/a)
        elif self.front_danger_action or self.back_danger_action:
            r_safec = -2
        elif self.front_close_danger_action or self.back_close_danger_action :
            r_safec = -5
        else:
            r_safec = 0

        #comf_rewards
        r_comf = -0.1*abs(jerk)
        r_speed = -0.1 * abs(new_speed - Ego().speedLimit)
        # print('r_dest: {} r_safe:{} r_safec:{} r_speed:{} rcom:{}'.format(r_dest, r_safe, r_safec, r_speed,r_comf))
        # return float((w_speed*r_speed+ w_comf* r_comf + w_safe*r_safe+w_dest*r_dest)/w_sum)
        return float(w_speed * r_speed + w_comf * r_comf + w_safe * r_safe +w_dest*r_dest+w_safec*r_safec)



    def getclosecar(self,vehicle,ego_point,ego_speed,num_front=2,num_back=2,action_constraint=False):
        before_car=[]
        after_car=[]
        vehicle_tuples=[]
        veh = []
        self.t2merge_front = []
        self.t2merge_back = []
        egodist2merge = self.distance(self.merge_pos, ego_point)
        for car in vehicle:
            carpoint = traci.vehicle.getPosition(car)
            if car == "ego" :
                continue
            if self.distance(carpoint,ego_point)>125:
                continue
            else:
                self.car_speed=traci.vehicle.getSpeed(car)

                cardist2merge = self.distance(self.merge_pos,carpoint)
                if action_constraint == False:
                    # if dist < 5:
                    if abs(ego_point[0] - carpoint[0]) < 5 and abs(ego_point[1]-carpoint[1]) < 1.8 :
                        self.is_danger = True

                car_x=carpoint[0]
                ego_x=ego_point[0]
                if ego_x > car_x:
                    # before_car.append(((ego_speed-car_speed)/35,(ego_x - car_x)/200))
                    before_car.append(((ego_speed * self.cos - self.car_speed)/30, (ego_point[0] - carpoint[0])/125, (ego_point[1] - carpoint[1])/125))
                    self.t2merge_front.append((cardist2merge/(self.car_speed+1e-5), ego_point[0] - carpoint[0]))
                else:
                    # after_car.append(((ego_speed-car_speed)/35,(ego_x - car_x)/200))
                    after_car.append(((ego_speed*self.cos - self.car_speed)/30, (ego_point[0] - carpoint[0])/125, (ego_point[1] -carpoint[1])/125))
                    self.t2merge_back.append((cardist2merge/(self.car_speed+1e-5), ego_point[0] - carpoint[0]))
        before_car.sort(key=lambda x: abs(x[1]))
        after_car.sort(key=lambda x: abs(x[1]))
        self.t2merge_front.sort(key=lambda x: abs(x[1]))
        self.t2merge_back.sort(key=lambda x: abs(x[1]))
        while len(before_car) < num_front:
            before_car.append((0, 0, 0))
            self.t2merge_front.append((0,0))
        while len(after_car) < num_back:
            after_car.append((0, 0, 0))
            self.t2merge_back.append((0,0))
        # print('before_car:{}  after:{}'.format(before_car,after_car))
        vehicle_tuples.extend(before_car[:num_front])
        vehicle_tuples.extend(after_car[:num_back])
        self.t2merge_front = self.t2merge_front[:1]
        self.t2merge_back = self.t2merge_back[:1]
        if ego_speed == 0 :
            self.tego2merge = np.inf
        else:
            self.tego2merge = egodist2merge/ego_speed
        # print('t2merge_front:{} t2merge_back:{} ego:{}'.format(self.t2merge_front,self.t2merge_back,self.tego2merge))
        for i in range(len(vehicle_tuples)):
            veh += vehicle_tuples[i]

        return veh

    def is_done(self):
        done = False
        # colli_num=traci.simulation.getCollidingVehiclesNumber()
        if self.crash:
            done = True
        if self.egoID not in self.veh_allList:
            done = True
        if self.timestep > self.max_timesteps:
            done = True
        if self.is_danger:
            done = True
        if self.is_final_success:
            done = True
        return done



    def preStep(self):
        # for i in range(2):
        traci.simulationStep()
        self.veh_allList = traci.vehicle.getIDList()

    def get_TTC(self,action_constraint):
        if action_constraint==True:
            front_car_point_diff = self.FalsevehList[1]*125
            back_car_point_diff = -self.FalsevehList[7]*125
            front_speed_diff = self.FalsevehList[0]*30
            back_speed_diff = -self.FalsevehList[6]*30
        else:
            front_car_point_diff = self.vehicle_tuple[1]*125
            back_car_point_diff = -self.vehicle_tuple[7]*125
            front_speed_diff = self.vehicle_tuple[0]*30
            back_speed_diff = -self.vehicle_tuple[6]*30
        front_distance_diff = front_car_point_diff - Ego().ego_length
        if front_distance_diff < 0:
            TTC_front = 0.01
        elif front_speed_diff > 0 and not np.isinf(front_distance_diff):
            TTC_front = front_distance_diff / front_speed_diff
        else:
            TTC_front = np.inf

        back_distance_diff = back_car_point_diff - Ego().ego_length
        if back_distance_diff < 0:
            TTC_back = 0.01
        elif back_speed_diff > 0 and not np.isinf(back_distance_diff):
            TTC_back = back_distance_diff / back_speed_diff
        else:
            TTC_back = np.inf
        return TTC_front,TTC_back,front_distance_diff,back_distance_diff,front_speed_diff,back_speed_diff








