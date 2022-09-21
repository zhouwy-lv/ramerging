import math
import os

import cv2
import gym
import numpy as np
from gym import spaces
import torch
from nn_builder.pytorch.NN import NN
from torch.distributions import Normal
from openpyxl import Workbook

import environments
from torch.distributions.utils import _standard_normal, broadcast_all
import traci
hyperparameters = {
    "learning_rate": 3e-4,
    "linear_hidden_units": [256] * 2,
    "final_layer_activation": None,
    "batch_norm": False,
    "tau": 0.001,  # 0.005
    "gradient_clipping_norm": 3,
    "initialiser": "Xavier"}
EPSILON = 1e-6

# resume_path ='E:\SUMO_PY\SUMO_RampMerge_1\\results\Models\RampMerging-v1\DDPG\\20210428222411\DDPG_3700_682.model'


def create_NN(input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
    """Creates a neural network for the agents to use"""
    default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                      "initialiser": "he", "batch_norm": False,
                                      "columns_of_data_to_be_embedded": [],
                                      "embedding_dimensions": [], "y_range": ()}

    for key in default_hyperparameter_choices:
        if key not in hyperparameters.keys():
            hyperparameters[key] = default_hyperparameter_choices[key]

    return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
              output_activation=hyperparameters["final_layer_activation"],
              batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
              hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
              columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
              embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
              random_seed=1).to(torch.device('cuda'))


def get_action_size(env):
    """Gets the action_size for the gym env into the correct shape for a neural network"""
    # if "overwrite_action_size" in self.config.__dict__: return self.config.overwrite_action_size
    if "action_size" in env.__dict__: return env.action_size

    else: return env.action_space.shape[0]

def distance (point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def load_q_network(input_dim, output_dim):
    save = torch.load(resume_path)
    resume_dict = save['actor_network_local']
    net = create_NN(input_dim, output_dim, hyperparameters=hyperparameters)
    net.load_state_dict(resume_dict, strict=True)
    net.eval()
    return net

def get_action(net, state):
    state = torch.from_numpy(state).cuda().unsqueeze(dim=0).float()
    with torch.no_grad():
        action = net(state)
        # action_idx = out.argmax(dim=-1).item()
    return action
#sac
def produce_action_and_action_info(net, state,action_size):
    """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
    actor_output = get_action(net,state)
    mean, log_std = actor_output[:, :action_size], actor_output[:, action_size:]
    std = log_std.exp()
    normal = Normal(mean, std)
    x_t = normal.rsample()  #rsample means it is sampled using reparameterisation trick
    action = torch.tanh(x_t)

    log_prob = normal.log_prob(x_t)
    log_prob -= torch.log(1 - action.pow(2) + EPSILON)
    log_prob = log_prob.sum(1, keepdim=True)
    return action, log_prob, torch.tanh(mean)

def xlsx_save(data, path, header=False):
    if len(data.shape) == 1: data = data[:, np.newaxis]
    wb = Workbook()
    ws = wb.active
    if header: ws.append(header)
    [h, l] = data.shape  # h:rows，l:columns
    for i in range(h):
        row = []
        for j in range(l):
            row.append(data[i,j])
        ws.append(row)
    wb.save(path)


if __name__ == '__main__':

    env = gym.make('RampMerging-v1')
    # env.test()
    success = []
    finalsuccess = []
    timeout = []
    fail = []

    data_results_root = os.path.join(os.path.dirname(__file__) + "/data_and_graphs/dataMediumn_2")
    # os.makedirs(data_results_root)
    # path = 'C:\\Users\zwy\Desktop\论文\\backspeed.xlsx'
    # path1 = 'C:\\Users\zwy\Desktop\论文\\distance.xlsx'
    # file_handle = open('evaluate.txt', mode='a')
    # file_handle.write('path:{} \n'.format(resume_path))
    # file_handle.close()
    state = env.reset()
    merge_point = traci.junction.getPosition('highway_out_j')

    action_size = int(get_action_size(env))
    net = load_q_network(input_dim=state.size, output_dim=action_size)
    total_reward = 0.
    for epi in range(2):
        for i in range(10):
            backcar = []
            frontcar = []
            backcar_speed = []
            dist2fap = []
            density=[]
            timestep = 0

            # tic1 = time.time()
            while True:
                # traci.gui.screenshot('View #0','text.jpg',width=-10,height=-10)

                # action,_,_ = produce_action_and_action_info(net, state,action_size)
                carnum = 0
                action = get_action(net,state)
                action = action.detach().cpu().numpy()[0]
                state, reward, done, info = env.step(action[0])
                timestep += 1
                total_reward += reward


                ego_p=traci.vehicle.getPosition("ego")
                cos = (abs(ego_p[0] - merge_point[0])) / distance(ego_p,merge_point)
                ego_speed = traci.vehicle.getSpeed("ego")
                ego2merge = distance(ego_p, merge_point)
                cos = (abs(ego_p[0] - merge_point[0])) / ego2merge
                if abs(ego2merge) < 100:
                    frontcar=[]
                    backcar=[]
                    vehlist = traci.vehicle.getIDList()
                    for car in vehlist:
                        car_point=traci.vehicle.getPosition(car)
                        car2merge = distance(car_point,merge_point)
                        car_speed = traci.vehicle.getSpeed(car)
                        if car == "ego":
                            continue
                        if distance(ego_p,car_point) > 125:
                            continue
                        if (ego_p[0]-merge_point[0])>0:
                            if(car_point[0]-merge_point[0])>0:
                                if (ego2merge - car2merge) > 0:
                                    frontcar.append((ego2merge-car2merge,car_speed))
                                else:
                                    backcar.append((car2merge-ego2merge, car_speed))
                            else:
                                frontcar.append((ego2merge+car2merge,car_speed))
                        else:
                            if (car_point[0]-merge_point[0])<0:
                                if (ego2merge - car2merge) > 0:
                                    backcar.append((ego2merge-car2merge, car_speed))
                                else:
                                    frontcar.append((car2merge-ego2merge, car_speed))
                            else:
                                backcar.append((ego2merge+car2merge,car_speed))
                    backcar.sort(key=lambda x: abs(x[0]))
                    frontcar.sort(key=lambda x: abs(x[0]))
                    while len(backcar) < 2:
                        backcar.append((0, 0))
                    while len(frontcar) < 2:
                        frontcar.append((0, 0))
                    backcar_speed.append((backcar[0][1],backcar[1][1],ego_speed,ego2merge))
                    dist2fap.append((backcar[0][0],backcar[1][0],frontcar[0][0],ego2merge))
                    a=backcar[0][0]
                    b=backcar[1][0]



                if done:
                    state = env.reset()
                    print('total_reward:', total_reward)
                    total_reward = 0.
                    success.append(info['is_success'])
                    finalsuccess.append(info['is_final_sccess'])
                    timeout.append(info['time_out'])
                    fail.append(info['is_danger'])

                    path_speed = os.path.join(data_results_root, "speeddata_{0}{1}.xlsx".format(epi,i))
                    path_dist = os.path.join(data_results_root, "distdata_{0}{1}.xlsx".format(epi,i))
                    xlsx_save(np.array(backcar_speed), path_speed)
                    xlsx_save(np.array(dist2fap), path_dist)
                    backcar = []
                    frontcar = []
                    backcar_speed = []
                    dist2fap = []
                    break
            print(i)
        print('epi:{} success_mean:{} finalsuccessmean:{} timeoutmean:{} failmean:{}'.format(epi,np.mean(success),np.mean(finalsuccess),np.mean(timeout),np.mean(fail)))
        # file_handle = open('evaluate.txt', mode='a')
        #
        # file_handle.write('tfc=0 epi:{} success_mean:{} finalsuccessmean:{} timeoutmean:{} failmean:{} \n'.format(epi,np.mean(success),np.mean(finalsuccess),np.mean(timeout),np.mean(fail)))
        # file_handle.close()
        success = []
        finalsuccess = []
        timeout = []
        fail = []
    pass


