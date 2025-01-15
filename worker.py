import threading
from queue import Queue
import logging 

import jax
from jax import jit, vmap
import time
from collections import deque
import numpy as np
import sys
from flax import serialization
import os
from running_mean_std_jax import RunningMeanStd

np.set_printoptions(threshold=sys.maxsize)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.basicConfig(
    format='%(asctime)s [%(threadName)s] %(levelname)s: %(message)s',
    level=logging.WARNING
)

class WorkerThread(threading.Thread):
    def __init__(self, config, env, rng_key, result_queue: Queue, agent_queue:Queue, agent_config, name='Worker Thread'):
        super(WorkerThread, self).__init__(name=name)
        self.daemon = True
        self.config = config
        self.agent_config = agent_config
        
        self.output_queue = result_queue
        self.agent_queue = agent_queue
        self.running = False
        self.device = config['worker_device']
        self.env_batch_size = config['worker_batch_size']
        self.random_steps_count = config['random_steps_count']

        self.validant_last_reward = None
        self.validant_best_reward = None
        
        self.validation_steps = 2000
        self.validation_start = 2000
        
        self.sims_per_validation_agent = 50
        
        self.step = 0
        self.env = env
        
        validation_agents_count = len(self.agent_config['validation_agent_params'])
        validation_batch_size = validation_agents_count * self.sims_per_validation_agent
        self.total_batch_size = self.env_batch_size + validation_batch_size
        
        logger.debug(f'Total batch size: {self.total_batch_size}, trainee batch size: {self.env_batch_size}, validation batch size: {validation_batch_size}')
        
        self._episode_rewards = jax.device_put(jax.numpy.zeros((self.env_batch_size, 1)), device=self.device)
        
        self.rng_key, key0 = jax.device_put(jax.random.split(rng_key, 2), device=self.device)
        
        self.env_rng_key = jax.device_put(jax.random.split(key0, self.total_batch_size), device=self.device)
        
        self.batched_reset = vmap(self.env.reset)
        self.batched_step = jit(vmap(self.env.step))
        
        self.jit_agent_double_step = jit(self.agent_double_step)
        
        self.jit_random_step = jit(self.random_step)
        self.jit_reset = jit(self.batched_reset)
        
        self.normalize_obs = True
        
        self.discard_agent0_data = False
        self.discard_agent1_data = False
        
        self.running_obs_mean_std = RunningMeanStd(shape=self.env.observation_space_shape)
        
        if self.normalize_obs:
            obs_norm_values = np.load('obs-norm/obs_norm_legacy.npz')
            self.obs_rms_mean = jax.device_put(jax.numpy.array(obs_norm_values['obs_mean']), device=self.device)
            self.obs_rms_var = jax.device_put(jax.numpy.array(obs_norm_values['obs_var']),  device=self.device)
            logging.debug('obs mean', self.obs_rms_mean)
            logging.debug('obs var', self.obs_rms_var)
            self.obs_rms_mean = jax.numpy.tile(self.obs_rms_mean, (self.total_batch_size, 1))
            self.obs_rms_var = jax.numpy.tile(self.obs_rms_var, (self.total_batch_size, 1))
        
        self.episode_reward = deque(maxlen=5)
        
        
    def random_step(self, env_state, inference_key):
        k = 1.00
        
        agent0_obs = self.normalize_obs_if_needed(env_state['obs'][0])
        agent1_obs = self.normalize_obs_if_needed(env_state['obs'][1])
        
        action0 = jax.random.uniform(inference_key, 
                                     (self.total_batch_size, self.env.action_space_shape[0]),
                                     minval=k*self.env.ctrlrange_low, maxval=k*self.env.ctrlrange_high)
        
        action1 = jax.random.uniform(inference_key, 
                                     (self.total_batch_size, self.env.action_space_shape[0]),
                                     minval=k*self.env.ctrlrange_low, maxval=k*self.env.ctrlrange_high)
        actions = (action0, action1)
        combined_actions = jax.numpy.concat(actions, axis=1)
        state, true_obs, true_reward, terminated, validation_rewards  = self.batched_step(env_state, combined_actions)
        
        agent0_true_obs = self.normalize_obs_if_needed(true_obs[0])
        agent1_true_obs = self.normalize_obs_if_needed(true_obs[1])
        
        validation_terminated = 0
        validation_reward = 0
        return state, tuple(actions), (agent0_obs, agent1_obs), (agent0_true_obs, agent1_true_obs), true_reward, terminated, validation_terminated, validation_reward
    
        
    def agent_double_step(self, env_state, inference_key, trainee_params, validant_params, *opponent_params):
        val_agent_params = self.agent_config['validation_agent_params']
        opponen_count = len(opponent_params)
        val_count = len(val_agent_params)
        ref_count = len(self.agent_config['ref_agent_params'])
        
        train_key, validation_key = jax.random.split(inference_key, 2)
        
        train_keys = jax.random.split(train_key, opponen_count+1)
        opponent_keys = train_keys[1:]
        
        validation_keys = jax.random.split(validation_key, val_count+1)
        val_keys = validation_keys[1:]
        
        trainee_func = self.agent_config['trainee_func']
        ref_agent_func = self.agent_config['ref_agent_func']
        val_agent_func = self.agent_config['validation_agent_func']
        
        agent0_obs = self.normalize_obs_if_needed(env_state['obs'][0])
        agent1_obs = self.normalize_obs_if_needed(env_state['obs'][1])
        
        action0, _, _ = trainee_func(trainee_params, agent0_obs[:self.env_batch_size], train_keys[0])
        action_validant, _, _ = trainee_func(validant_params, agent0_obs[self.env_batch_size:], validation_keys[0])
        
        opponent_obs_size = self.env_batch_size // opponen_count
        opponent_actions = []
        opponent_obs_ptr = 0
        
        for i in range(opponen_count):
                next_obs_ptr = opponent_obs_ptr + opponent_obs_size
                current_opponent_obs =  agent1_obs[opponent_obs_ptr:next_obs_ptr] if i < opponen_count-1 else agent1_obs[opponent_obs_ptr:self.env_batch_size]
                #print(f'obs[{opponent_obs_ptr}:{next_obs_ptr}]')
                if i < ref_count:
                    opponent_action, _, _ = ref_agent_func(opponent_params[i], current_opponent_obs, opponent_keys[i])
                else:
                    opponent_action, _, _ = trainee_func(opponent_params[i], current_opponent_obs, opponent_keys[i])
                opponent_actions.append(opponent_action)
                opponent_obs_ptr = next_obs_ptr
                
        val_obs_ptr = self.env_batch_size 
        for i in range(val_count):
                next_obs_ptr = val_obs_ptr + self.sims_per_validation_agent
                current_opponent_obs =  agent1_obs[val_obs_ptr:next_obs_ptr] if i < val_count-1 else agent1_obs[val_obs_ptr:]
                #print(f'obs[{opponent_obs_ptr}:{next_obs_ptr}]')
                opponent_action, _, _ = val_agent_func(val_agent_params[i], current_opponent_obs, val_keys[i])
                opponent_actions.append(opponent_action)
                val_obs_ptr = next_obs_ptr
            
        actions = (jax.numpy.concatenate((action0, action_validant), axis=0), 
                   jax.numpy.concatenate(opponent_actions, axis=0))
        
        state, true_obs, true_reward, terminated, validation_rewards = self.batched_step(env_state, jax.numpy.concat(actions, axis=1))
        
        agent0_true_obs = self.normalize_obs_if_needed(true_obs[0])
        agent1_true_obs = self.normalize_obs_if_needed(true_obs[1])
        
        validation_terminated = jax.numpy.sum(terminated[self.env_batch_size:])
        validation_reward = jax.numpy.sum(validation_rewards[0][self.env_batch_size:])
        return state, actions, (agent0_obs, agent1_obs), (agent0_true_obs, agent1_true_obs), true_reward, terminated, validation_terminated, validation_reward
        
    
    def perform_step(self, state0, step_fun, agent_params=None):
        inference_key, self.rng_key = jax.random.split(self.rng_key, 2)

        self.running_obs_mean_std.update(state0['obs'][0])
        self.running_obs_mean_std.update(state0['obs'][1])
        
        state, actions, obs0, obs1, rewards, terminated, validation_terminated, validation_reward = step_fun(state0, inference_key, *agent_params) if agent_params is not None else step_fun(state0, inference_key)
        
        self.compute_metrics(rewards, terminated)
        
        obss_cpu = [jax.device_get(obs) for obs in obs0]
        next_obs_cpu = [jax.device_get(t_obs) for t_obs in obs1]
        reward_cpu = [jax.device_get(t_rew) for t_rew in rewards]
        action_cpu = jax.device_get(actions)
        terminated_cpu = jax.device_get(terminated)

        if not self.discard_agent0_data:
            self.output_queue.put((obss_cpu[0], 
                                   next_obs_cpu[0],
                                   action_cpu[0],
                                   reward_cpu[0], 
                                   terminated_cpu))
        
        if not self.discard_agent1_data:
            self.output_queue.put((obss_cpu[1],
                                   next_obs_cpu[1],
                                   action_cpu[1],
                                   reward_cpu[1],
                                   terminated_cpu.copy()))
        return state, validation_terminated, validation_reward
    
    
    def compute_metrics(self, rewards, terminated):
        self._episode_rewards += rewards[0][:self.env_batch_size]
        terminated_count = jax.numpy.sum(terminated[:self.env_batch_size]).item()
        if terminated_count > 0:
            episode_reward = jax.numpy.sum(self._episode_rewards * terminated[:self.env_batch_size]).item() / terminated_count
            self.episode_reward.append(episode_reward)
            self._episode_rewards *= (1 - terminated[:self.env_batch_size])
     
    def write_params_to_file(self, filename, params):
        with open(filename, 'wb') as f:
            f.write(serialization.to_bytes(params))        
    
    def save_state(self, path, agent_params, qf1_params, qf2_params, q_opt_state, p_opt_state, alpha_opt_state):
        os.makedirs(path, exist_ok=True)
        self.write_params_to_file(os.path.join(path, 'agent.flax'), agent_params)
        self.write_params_to_file(os.path.join(path, 'q1.flax'), qf1_params)
        self.write_params_to_file(os.path.join(path, 'q2.flax'), qf2_params)
        self.write_params_to_file(os.path.join(path, 'q_opt.flax'), q_opt_state)
        self.write_params_to_file(os.path.join(path, 'p_opt.flax'), p_opt_state)
        self.write_params_to_file(os.path.join(path, 'a_opt.flax'), alpha_opt_state)     
    
    def transfer_params_if_needed(self, params):
            if self.device != self.config['trainer_device']:
                return jax.device_put(params, self.device)
            return params
    
    def run(self):
        use_norm = '(with obs norm)' if self.normalize_obs else ''
        logger.info(f"MultiSAC-MJX {use_norm} {self.name} alive on {self.device}!")
        self.running = True

        state = self.jit_reset(self.env_rng_key)
        state['step'] = jax.random.randint(self.rng_key, state['step'].shape, 0, self.env.max_steps-2) 
        
        for i in range(self.random_steps_count):
            state, validation_terminated, validation_reward = self.perform_step(state, self.jit_random_step)
            if i % 25 == 0 and len(self.episode_reward)>0:
                episode_reward = jax.numpy.array(self.episode_reward).mean().item()
                logger.info(f'{i:05d}\t episode_reward: {episode_reward:1.4f}')
                
        logger.info('Random steps finised')
        
        self.step = 0
        last_agents = deque(maxlen=self.config["last_agents_count"])
        
        _, params, qf1, qf2, _, opt_params = self.agent_queue.get()
        params = self.transfer_params_if_needed(params)
        ref_agents = self.agent_config["ref_agent_params"]
        validant_params = params
        validant_qf1 = qf1
        validant_qf2 = qf2
        validant_opt_params = opt_params 
                
        for _ in range(self.config["last_agents_count"]):
            last_agents.append(params) 
           
        logger.debug('Starting worker loop')
        total_validation_reward = 0.0
        total_episodes = 0
        validation_step = 0
        
        while self.running: 
            trainee_name, trainee_params, qf1, qf2, global_step, opt_params = self.agent_queue.get()
            trainee_params = self.transfer_params_if_needed(trainee_params)
            
            if self.step % self.config['self_play_lag'] == 0:
                    _, p, _, _, _, _ = self.agent_queue.get()
                    last_agents.append(self.transfer_params_if_needed(p))
               
            state, validation_terminated, validation_reward = self.perform_step(state, self.jit_agent_double_step, (trainee_params, validant_params, *list(ref_agents), *list(last_agents)))
            
            if self.step % 5000 == 0:
                mean = jax.device_get(self.running_obs_mean_std.mean)
                var = jax.device_get(self.running_obs_mean_std.var)
                np.savez('obs-norm/obs_norm_last.npz',obs_mean=mean, obs_var=var)
            self.step += 1 
            
            if self.step==self.validation_start:
                logger.debug(f'Starting agent validation ({self.validation_steps} steps, {self.sims_per_validation_agent} sims per validating agent, {len(self.agent_config["validation_agent_params"])} validation agents)')
            
            if self.step>=self.validation_start:
                total_validation_reward += validation_reward
                total_episodes += validation_terminated
                reward_per_episode = total_validation_reward/total_episodes
                validation_step += 1  
            
            if validation_step >= self.validation_steps:
                self.validant_last_reward = reward_per_episode
                
                if self.validant_best_reward is None or reward_per_episode > self.validant_best_reward:
                    self.validant_best_reward = reward_per_episode

                    agent_path=os.path.join('checkpoints', f'{trainee_name}', f'{global_step}')             
                    logger.info(f"{global_step} new best agent: {reward_per_episode:1.2f} -> '{agent_path}'")
                    self.save_state(agent_path, validant_params, validant_qf1, validant_qf2, *validant_opt_params)
                    
                validant_params = trainee_params
                validant_opt_params = opt_params
                validant_qf1 = qf1
                validant_qf2 = qf2
                total_validation_reward = 0.0
                total_episodes = 0
                validation_step = 0
                
            
        logger.debug(f"Worker finished")
    
    def normalize_obs_if_needed(self, obs):
        if not self.normalize_obs:
            return obs
        return jax.numpy.clip((obs - self.obs_rms_mean) / jax.numpy.sqrt(self.obs_rms_var + 1e-8), -10.0, 10.0)
