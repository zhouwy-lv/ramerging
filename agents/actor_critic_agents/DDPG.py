import glob
import os
import re

import numpy as np
import torch
import torch.nn.functional as functional
from torch import optim
from agents.Base_Agent import Base_Agent
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration

class DDPG(Base_Agent):
    """A DDPG Agent"""
    agent_name = "DDPG"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        Base_Agent.copy_model_over(self.actor_local, self.actor_target)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.exploration_strategy = OU_Noise_Exploration(self.config)
        # self.actor_loss=[]
        # self.critic_loss=[]
        # self.agent_name = agent_name
        if config.resume:
            self.load_resume(config.resume_path)

    def step(self):
        """Runs a step in the game"""

        while not self.done:
            # print("State ", self.state.shape)
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                self.actor_loss = []
                self.critic_loss = []
                for i in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    states, actions, rewards, next_states, dones = self.sample_experiences()
                    self.critic_learn(states, actions, rewards, next_states, dones,i)
                    self.actor_learn(states,i)
                self.iteration_number=np.int(self.global_step_number/self.hyperparameters["update_every_n_steps"])
                print('\n iteration:{} episode:{} ACtloss:{} CRIloss:{}'.format(self.iteration_number,self.episode_number,
                                                                                          np.mean(self.actor_loss),np.mean(self.critic_loss)))
                self.writer.add_scalar('loss/{}_actloss'.format(self.agent_name), np.mean(self.actor_loss), self.iteration_number)
                self.writer.add_scalar('loss/{}_crtloss'.format(self.agent_name), np.mean(self.critic_loss), self.iteration_number)
            self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1
        return self.agent_name

    def sample_experiences(self):
        return self.memory.sample()

    def pick_action(self, state=None):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""

        if state is None:

            # self.state.astype(float)
            state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": action})
        return action.squeeze(0)

    def critic_learn(self, states, actions, rewards, next_states, dones,i):
        """Runs a learning iteration for the critic"""
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        losses=loss.cpu().detach().numpy()
        self.critic_loss.append(float(losses))
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, loss, self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss for the critic"""
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        critic_expected = self.compute_expected_critic_values(states, actions)
        loss = functional.mse_loss(critic_expected, critic_targets)
        return loss

    def compute_critic_targets(self, next_states, rewards, dones):
        """Computes the critic target values to be used in the loss for the critic"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)
        return critic_targets

    def compute_critic_values_for_next_states(self, next_states):
        """Computes the critic values for next states to be used in the loss for the critic"""
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            critic_targets_next = self.critic_target(torch.cat((next_states, actions_next), 1))
        return critic_targets_next

    def compute_critic_values_for_current_states(self, rewards, critic_targets_next, dones):
        """Computes the critic values for current states to be used in the loss for the critic"""
        critic_targets_current = rewards + (self.hyperparameters["discount_rate"] * critic_targets_next * (1.0 - dones))
        return critic_targets_current

    def compute_expected_critic_values(self, states, actions):
        """Computes the expected critic values to be used in the loss for the critic"""
        critic_expected = self.critic_local(torch.cat((states, actions), 1))
        return critic_expected

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def actor_learn(self, states,i):
        """Runs a learning iteration for the actor"""
        # if self.done: #we only update the learning rate at end of each episode
        #     self.update_learning_rate(self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer)
        actor_loss = self.calculate_actor_loss(states)
        actor_losses=actor_loss.cpu().detach().numpy()
        self.actor_loss.append(float(actor_losses))
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.hyperparameters["Actor"]["tau"])


    def calculate_actor_loss(self, states):
        """Calculates the loss for the actor"""
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()
        return actor_loss

    def locally_save_policy(self, best=True, episode=None):
        state = {'episode': self.episode_number,
                     'critic_network_local': self.critic_local.state_dict(),
                     'critic_network_target': self.critic_target.state_dict(),
                     'actor_network_local': self.actor_local.state_dict(),
                     'actor_network_target': self.actor_target.state_dict()
                 }
        # else:
        #     state = {'episode': self.episode_number,
        #              'q_network_local': self.q_network_local.state_dict()}

        model_root = os.path.join('Models', self.config.env_title, self.agent_name, self.config.log_base)
        if not os.path.exists(model_root):
            os.makedirs(model_root)

        if best:
            last_best_file = glob.glob(os.path.join(model_root, 'rolling_score*'))
            if last_best_file:
                os.remove(last_best_file[0])

            save_name = model_root + "/rolling_score_%.4f.model"%(self.rolling_results[-1])
            torch.save(state, save_name)
            self.logger.info('Model-%s save success...' % (save_name))
        else:
            save_name = model_root + "/%s_%d_%d.model" % (self.agent_name, self.episode_number,self.iteration_number)
            torch.save(state, save_name)
            self.logger.info('Model-%s save success...' % (save_name))

    def load_resume(self, resume_path):
        save = torch.load(resume_path)
        # if self.agent_name != "DQN":
        critic_network_local_dict = save['critic_network_local']
        critic_network_target_dict = save['critic_network_target']
        self.critic_local.load_state_dict(critic_network_local_dict, strict=True)
        self.critic_target.load_state_dict(critic_network_target_dict, strict=True)
        actor_network_local_dict = save['actor_network_local']
        actor_network_target_dict = save['actor_network_target']
        self.actor_local.load_state_dict(actor_network_local_dict, strict=True)
        self.actor_target.load_state_dict(actor_network_target_dict, strict=True)
        # else:
        #     q_network_local_dict = save['q_network_local']
        #     self.q_network_local.load_state_dict(q_network_local_dict, strict=True)
        self.logger.info('load resume model success...')

        file_name = os.path.basename(resume_path)
        if self.agent_name == "TD3":
            episode_str_epi = re.findall(r"\d+\.?\d*", file_name)[1]
            episode_str_iter = re.findall(r"\d+\.?\d*", file_name)[2]
        else:
            episode_str_epi = re.findall(r"\d+\.?\d*", file_name)[0]
            episode_str_iter = re.findall(r"\d+\.?\d*", file_name)[1]
        episode_iter_list = episode_str_iter.split('.')
        if not episode_iter_list[1]:
            episode_iter = episode_iter_list[0]
        else:
            episode_iter = 0

        if not self.config.retrain:
            self.episode_number = int(episode_str_epi)
            self.iteration_number = int(episode_iter)
            self.global_step_number = self.iteration_number*self.hyperparameters["update_every_n_steps"]
        else:
            self.episode_number = 0
            self.iteration_number = 0