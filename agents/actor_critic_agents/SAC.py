import glob
import os
import re

from tensorboardX import SummaryWriter

from agents.Base_Agent import Base_Agent
from utilities.OU_Noise import OU_Noise
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6

class SAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "CONTINUOUS", "Action types must be continuous. Use SAC Discrete instead for discrete actions"
        assert self.config.hyperparameters["Actor"]["final_layer_activation"] != "Softmax", "Final actor layer must not be softmax"
        self.hyperparameters = config.hyperparameters
        self.writer = SummaryWriter('./log')
        self.reward_scale = 1
        self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                            key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size * 2, key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(self.device)).item() # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4) #1e-4
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

        if config.resume:
            self.load_resume(config.resume_path)

    def save_result(self):
        """Saves the result of an episode of the game. Overriding the method in Base Agent that does this because we only
        want to keep track of the results during the evaluation episodes"""
        if self.episode_number == 1 or not self.do_evaluation_iterations:
            self.game_full_episode_scores.extend([self.total_episode_score_so_far])
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
            self.save_max_result_seen()

        elif (self.episode_number - 1) % TRAINING_EPISODES_PER_EVAL_EPISODE == 0:
            self.game_full_episode_scores.extend([self.total_episode_score_so_far for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.rolling_results.extend([np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]) for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.save_max_result_seen()
            # self.iteration_number +=1

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        if self.add_extra_noise: self.noise.reset()

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        # self.qf1los=[]
        # self.qf2los=[]
        # self.actor_loss=[]
        # self.alphaloss = []
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)

            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                self.qf1los = []
                self.qf2los = []
                self.actor_loss = []
                self.alphaloss = []
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
                self.iteration_number=np.int(self.global_step_number/1024)
                print('\n iteration:{} episode:{} qloss:{} policyloss:{} alphaloss:{}'.format(self.iteration_number,self.episode_number,
                                                                                          np.mean(self.qf1los),np.mean(self.qf2los),np.mean(self.alphaloss)))
                self.writer.add_scalar('loss/SAC_qf1loss', np.mean(self.qf1los), self.iteration_number)
                self.writer.add_scalar('loss/SAC_qf2loss', np.mean(self.qf2los), self.iteration_number)
                self.writer.add_scalar('loss/SAC_actorloss', np.mean(self.actor_loss), self.iteration_number)
                self.writer.add_scalar('loss/SAC_alphaloss', np.mean(self.alphaloss), self.iteration_number)
            mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            # if self.info['discard_data'] == 1:
            #     print(' discard data')
            if not eval_ep :
                self.save_experience(experience=(self.state, self.action, self.reward*self.reward_scale, self.next_state, mask))
            self.state = self.next_state
            self.global_step_number += 1
        # print(self.total_episode_score_so_far)
        if eval_ep: self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1
        # if len(self.qf1los)==0 :
        #     self.qf1los.append(1)
        #     self.alphaloss.append(1)
        #     self.qf2los.append(1)
        #     self.actor_loss.append(1)
        # if self.automatic_entropy_tuning:
        #     return [np.mean(self.qf1los), np.mean(self.qf2los), np.mean(self.actor_loss),np.mean(self.alphaloss)], self.agent_name
        # else:
        return self.agent_name

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if state is None: state = self.state

        # print(state)
        if eval_ep:
            action = self.actor_pick_action(state=state, eval=True)
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            action = self.environment.action_space.sample()
            # print("Picking random action ", action)
        else: action = self.actor_pick_action(state=state)

        if self.add_extra_noise:
            action += self.noise.sample()

        return float(action)

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None: state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1: state = state.unsqueeze(0)
        if eval == False: action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()

        return action[0]

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  #rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
               self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        # state_batch_mean = torch.mean(state_batch, 0)
        # state_batch_std = torch.std(state_batch, 0)
        # state_batch = (state_batch-state_batch_mean)/(state_batch_std+1e-5)
        # next_state_batch_mean = torch.mean(next_state_batch, 0)
        # next_state_batch_std = torch.std(next_state_batch, 0)
        # next_state_batch = (next_state_batch-next_state_batch_mean)/(next_state_batch_std+1e-5)
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        self.update_critic_parameters(qf1_loss, qf2_loss)
        self.qf1los.append(float(qf1_loss.cpu().detach().numpy()))
        self.qf2los.append(float(qf2_loss.cpu().detach().numpy()))
        policy_loss, log_pi = self.calculate_actor_loss(state_batch)

        # max_val = next_state_batch.max()

        acloss=float(policy_loss.cpu().detach().numpy())

        if np.isinf(acloss):
            print('loss inf')
        if np.isnan(acloss):
            print('loss nan')
        self.actor_loss.append(float(policy_loss.cpu().detach().numpy()))
        if self.automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
            self.alphaloss.append(float(alpha_loss.cpu().detach().numpy()))

        else: alpha_loss = None
        self.update_actor_parameters(policy_loss, alpha_loss)


    def sample_experiences(self):
        return  self.memory.sample()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(torch.cat((next_state_batch, next_state_action), 1))
            qf2_next_target = self.critic_target_2(torch.cat((next_state_batch, next_state_action), 1))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)
        qf1 = self.critic_local(torch.cat((state_batch, action_batch), 1))
        qf2 = self.critic_local_2(torch.cat((state_batch, action_batch), 1))
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(torch.cat((state_batch, action), 1))
        qf2_pi = self.critic_local_2(torch.cat((state_batch, action), 1))

        state_val = state_batch.sum()
        cc = self.critic_local.state_dict()
        dd = self.critic_local_2.state_dict()
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        acloss=float(policy_loss.cpu().detach().numpy())
        if np.isnan(acloss):
            print('loss nan..')
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_critic_parameters(self, critic_loss_1, critic_loss_2):
        """Updates the parameters for both critics"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Updates the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        print("----------------------------")

    def locally_save_policy(self, best=True, episode=None):
        state = {'episode': self.episode_number,
                     'critic_network_local': self.critic_local.state_dict(),
                     'critic_network_target': self.critic_target.state_dict(),
                     'critic_local_2': self.critic_local_2.state_dict(),
                     'critic_target_2': self.critic_target_2.state_dict(),
                     'actor_local': self.actor_local.state_dict()
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
        actor_network_local_dict = save['actor_local']
        self.actor_local.load_state_dict(actor_network_local_dict, strict=True)
        critic_network_local2_dict = save['critic_local_2']
        critic_network_target2_dict = save['critic_target_2']
        self.critic_local_2.load_state_dict(critic_network_local2_dict, strict=True)
        self.critic_target_2.load_state_dict(critic_network_target2_dict, strict=True)
        # else:
        #     q_network_local_dict = save['q_network_local']
        #     self.q_network_local.load_state_dict(q_network_local_dict, strict=True)
        self.logger.info('load resume model success...')

        file_name = os.path.basename(resume_path)
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