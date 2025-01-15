import threading
from queue import Queue, Full
import logging 

from buffer import BufferThread
from q_network import SoftQNetwork_skip

import jax
from jax import jit, value_and_grad
import optax
import flax
from flax import traverse_util
from flax import serialization
import os

from worker import WorkerThread


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.basicConfig(
    format='%(asctime)s [%(threadName)s] %(levelname)s: %(message)s',
    level=logging.WARNING
)

class Reporter:
    def __init__(self, 
                 use_tensorboard: bool = False,
                 use_wandb: bool = False,
                 tensorboard_dir = "runs",
                 wandb_project = "sac-my-flax-arena-4",
                 wandb_config = None):
        
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.writer = None
        self.run_name = 'unnamed'
        self.wandb = None
        
        if use_wandb:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=wandb_project, config=wandb_config, sync_tensorboard=use_tensorboard)
            self.run_name = self.wandb.run.name
            
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(tensorboard_dir, self.run_name))

    def log(self, data: dict, step: int):
        # Log to TensorBoard
        if self.use_tensorboard:
            for key, value in data.items():
                self.writer.add_scalar(key, value, step)
                
        # Log to WandB
        if self.use_wandb and not self.use_tensorboard:
                self.wandb.log(data, step=step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.use_wandb:
            self.wandb.finish()
            

def linear_warmup_cosine_decay_scheduler(
    init_value, warmup_steps, decay_steps, end_value=None
):
    if end_value is None:
        end_value = init_value * 0.1  
    schedule = optax.join_schedules(
        [
            optax.linear_schedule(0.0, init_value, warmup_steps),
            optax.cosine_decay_schedule(init_value, decay_steps, end_value),
        ],
        [warmup_steps],
    )
    return schedule


class TrainingThread(threading.Thread):

    def __init__(self, config, buffer_thread: BufferThread, agent_queue: Queue, agent, agent_params, rng_key, observation_space_shape, action_space_shape, worker_thread: WorkerThread, name='Training Thread'):
        super(TrainingThread, self).__init__(name=name)
        self.daemon = True
        self.agent_queue = agent_queue
        self.running = False
        
        self.device = config['trainer_device']
        self.config = config
        self.batch_size = config['trainer_batch_size']
        self.agent = agent
        self.agent_params = jax.device_put(agent_params, self.device) 
        self.report_to_wandb = True
        
        self.buffer_thread = buffer_thread
        self.synchroneous = True
        
        self.initial_alpha = config['initial_alpha']
        self.tau = config['tau']
        self.gamma = config['gamma']
        self.q_lr = config['q_lr']
        self.p_lr = config['p_lr']
        self.policy_update_interval = 1
        self.autotune = config['autotune_alpha']
        self.total_steps=config['total_steps']
        
        self.warmup_steps = config['warmup_steps']
        
        self.qf1 = SoftQNetwork_skip()
        self.qf2 = SoftQNetwork_skip()
        
        self.qf1_target = SoftQNetwork_skip()
        self.qf2_target = SoftQNetwork_skip()
        
        init1, init2, self.rng_key = jax.device_put(jax.random.split(rng_key, 3), device=self.device)
        
        
        actions_count = jax.numpy.prod(jax.numpy.array(action_space_shape)).item()
        observations_count = jax.numpy.prod(jax.numpy.array(observation_space_shape)).item()
        
        dummy_obs = jax.numpy.ones((1, observations_count), device=self.device)
        dummy_action = jax.numpy.ones((1, actions_count), device=self.device)
        
        self.qf1_params = jax.jit(self.qf1.init)(init1, dummy_obs, dummy_action)['params']
        self.qf_target1_params = jax.jit(self.qf1.init)(init1, dummy_obs, dummy_action)['params']
        
        self.qf2_params = jax.jit(self.qf2.init)(init2, dummy_obs, dummy_action)['params']
        self.qf_target2_params = jax.jit(self.qf2.init)(init2, dummy_obs, dummy_action)['params']
        
        total_q_params = sum(p.size for p in jax.tree_util.tree_leaves(self.qf1_params))
        logger.info(f'Total number of parameters in the Q-network: {total_q_params}')
        
        total_agent_params = sum(p.size for p in jax.tree_util.tree_leaves(self.agent_params))
        logger.info(f'Total number of parameters in the agent: {total_agent_params}')
        
        decay_steps = self.total_steps - self.warmup_steps  
        self.p_lr_schedule = linear_warmup_cosine_decay_scheduler(self.p_lr, self.warmup_steps, decay_steps)
        self.q_lr_schedule = linear_warmup_cosine_decay_scheduler(self.q_lr, self.warmup_steps, decay_steps)
    
        self.q_optimizer = optax.nadam(learning_rate=self.q_lr_schedule)
        
        combined_params = {'qf1': self.qf1_params, 'qf2': self.qf2_params}
        self.q_opt_state = jax.jit(self.q_optimizer.init)(combined_params)
        
        self.p_optimizer = optax.nadam(learning_rate=self.p_lr_schedule)
        self.p_opt_state = jax.jit(self.p_optimizer.init)(self.agent_params)
        
        self.worker_thread = worker_thread
        
        self.target_entropy = -actions_count
        self.log_alpha = jax.device_put(jax.numpy.log(self.initial_alpha), device=self.device)
        self.alpha_optimizer = optax.nadam(learning_rate=self.p_lr_schedule)
        self.alpha_opt_state = jax.jit(self.alpha_optimizer.init)(self.log_alpha)
        self.alpha = jax.numpy.exp(self.log_alpha)
            
    def write_params_to_file(self, filename, params):
        with open(filename, 'wb') as f:
            f.write(serialization.to_bytes(params))
        
    def save_state(self, path):
        
        os.makedirs(path, exist_ok=True)
        
        self.write_params_to_file(os.path.join(path, 'agent.flax'), self.agent_params)
        self.write_params_to_file(os.path.join(path, 'q1.flax'), self.qf1_params)
        self.write_params_to_file(os.path.join(path, 'q2.flax'), self.qf2_params)
        self.write_params_to_file(os.path.join(path, 'q_opt.flax'), self.q_opt_state)
        self.write_params_to_file(os.path.join(path, 'p_opt.flax'), self.p_opt_state)
        self.write_params_to_file(os.path.join(path, 'a_opt.flax'), self.alpha_opt_state)     
            
    
    def load_state(self, path):
        logger.info(f'Loading state from {path}')
        with open(os.path.join(path,'agent.flax'), 'rb') as checkpoint:
            params_bytes = checkpoint.read()
            self.agent_params = serialization.from_bytes(self.agent_params, params_bytes)
            
        with open(os.path.join(path,'q1.flax'), 'rb') as checkpoint:
            params_bytes = checkpoint.read()
            self.qf1_params = serialization.from_bytes(self.qf1_params, params_bytes)
            self.qf_target1_params = serialization.from_bytes(self.qf_target1_params, params_bytes)
            
        with open(os.path.join(path,'q2.flax'), 'rb') as checkpoint:
            params_bytes = checkpoint.read()
            self.qf2_params = serialization.from_bytes(self.qf2_params, params_bytes)
            self.qf_target2_params = serialization.from_bytes(self.qf_target2_params, params_bytes)
            
        with open(os.path.join(path,'q_opt.flax'), 'rb') as checkpoint:
            params_bytes = checkpoint.read()
            self.q_opt_state = serialization.from_bytes(self.q_opt_state, params_bytes)

        with open(os.path.join(path,'p_opt.flax'), 'rb') as checkpoint:
            params_bytes = checkpoint.read()
            self.p_opt_state = serialization.from_bytes(self.p_opt_state, params_bytes)
            
        with open(os.path.join(path,'a_opt.flax'), 'rb') as checkpoint:
            params_bytes = checkpoint.read()
            self.alpha_opt_state = serialization.from_bytes(self.alpha_opt_state, params_bytes)

        
    def policy_grad(self, observations, agent_params, qf1_params, qf2_params, rng_key, alpha):
        
        def policy_loss(agent_params, qf1_params, qf2_params, observations):
            pi, log_pi, _ = self.agent.get_action(agent_params, observations, rng_key)
            
            qf1_pi = self.qf1.apply({'params': qf1_params}, observations, pi)
            qf2_pi = self.qf2.apply({'params': qf2_params}, observations, pi)
            combined_qf_pi = jax.numpy.minimum(qf1_pi, qf2_pi)   
            loss = jax.numpy.mean((alpha * log_pi) - combined_qf_pi)
            return loss, log_pi


        (p_loss, log_pi), p_grad = value_and_grad(policy_loss, argnums=0, has_aux=True)(agent_params, qf1_params, qf2_params, observations)
    
        return p_loss, log_pi, p_grad
    
    def alpha_step(self, log_alpha, log_pi):
        def alpha_loss_fn(log_alpha, log_pi):
            alpha = jax.numpy.exp(log_alpha)
            alpha_loss = -jax.numpy.mean(alpha * (log_pi + self.target_entropy))
            return alpha_loss
        alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha, log_pi)
        return alpha_loss, alpha_grad
    
    def q_grad(self, observations, next_observations, actions, rewards, dones,
               qf1_params, qf2_params, qf1_target_params, qf2_target_params, agent_params, 
               rng_key,
               alpha):
        next_state_actions, next_state_log_pi, _ = self.agent.get_action(agent_params, next_observations, rng_key)
        qf1_next_target = self.qf1_target.apply({'params': qf1_target_params}, next_observations, next_state_actions)
        qf2_next_target = self.qf2_target.apply({'params': qf2_target_params}, next_observations, next_state_actions)
        combined_qf_next_target = jax.numpy.minimum(qf1_next_target, qf2_next_target)
        combined_qf_next_target = combined_qf_next_target - alpha * next_state_log_pi
        d = (1 - dones)
        next_q_value = rewards + d * self.gamma * combined_qf_next_target
        
        def qf_loss(_qf1_params, _qf2_params):
            qf1_a_values = self.qf1.apply({'params': _qf1_params}, observations, actions)
            qf2_a_values = self.qf2.apply({'params': _qf2_params}, observations, actions) 
            
            qf1_loss = jax.numpy.square(qf1_a_values-next_q_value).mean() 
            qf2_loss = jax.numpy.square(qf2_a_values-next_q_value).mean() 
            return jax.numpy.mean(qf1_loss + qf2_loss)
        q_loss, grads = value_and_grad(qf_loss, argnums=(0, 1))(qf1_params, qf2_params)
        return q_loss, *grads
    

    def target_update(self, target_params, source_params, tau):
        flat_target_params = traverse_util.flatten_dict(target_params, sep='/')
        flat_source_params = traverse_util.flatten_dict(source_params, sep='/')
        
        updated_flat_params = {key: tau * flat_source_params[key] + (1 - tau) * flat_target_params[key]
                            for key in flat_target_params}
        
        unflat_params = traverse_util.unflatten_dict(updated_flat_params, sep='/')
        return flax.core.freeze(unflat_params)
    
    
    def q_step(self,
               q_opt_state,
               observations, next_observations, actions, rewards, dones,
               qf1_params, qf2_params, qf1_target_params, qf2_target_params, agent_params, rng_key, alpha):
        q_loss, qf1_grad, qf2_grad = self.q_grad(observations, 
                                                    next_observations,
                                                    actions,
                                                    rewards, 
                                                    dones,
                                                    qf1_params,
                                                    qf2_params,
                                                    qf1_target_params,
                                                    qf2_target_params,
                                                    agent_params,
                                                    rng_key,
                                                    alpha)
            
        combined_gradients = {'qf1': qf1_grad, 'qf2': qf2_grad}
        combined_params = {'qf1': qf1_params, 'qf2': qf2_params}
        
        updates, q_opt_state = self.q_optimizer.update(combined_gradients, q_opt_state, combined_params)
        updated_params = optax.apply_updates(combined_params, updates)
        return q_loss, updated_params['qf1'], updated_params['qf2'], q_opt_state
    
    def policy_step(self, 
                    p_opt_state, alpha_opt_state,
                    observations, 
                    agent_params,
                    new_qf1_params, new_qf2_params,
                    rng_key, alpha, log_alpha):
        for _ in range(self.policy_update_interval):
            key0, key1, rng_key = jax.random.split(rng_key, 3)
            p_loss, log_pi, p_grad = self.policy_grad(observations, agent_params, new_qf1_params, new_qf2_params, key0, alpha)
            
            p_updates, p_opt_state = self.p_optimizer.update(p_grad, p_opt_state, agent_params)
            agent_params = optax.apply_updates(agent_params, p_updates)
            
            if self.autotune:
                _, log_pi, _ = self.agent.get_action(agent_params, observations, key1)
                alpha_loss, alpha_grad = self.alpha_step(log_alpha, log_pi)
                updates, alpha_opt_state = self.alpha_optimizer.update(alpha_grad, alpha_opt_state)
                log_alpha = optax.apply_updates(log_alpha, updates)
                alpha = jax.numpy.exp(log_alpha)
        return p_loss, alpha_loss, agent_params, alpha, log_alpha, p_opt_state, alpha_opt_state
    
    def target_step(self, qf1_target_params, qf2_target_params, new_qf1_params, new_qf2_params):
        new_qf1_target1_params = self.target_update(qf1_target_params, new_qf1_params, self.tau)
        new_qf2_target1_params = self.target_update(qf2_target_params, new_qf2_params, self.tau)
        return new_qf1_target1_params, new_qf2_target1_params
    
    def run(self):
        logger.info(f"MultiSAC-MJX {self.name} alive on {self.device}!")
        self.running = True
        jit_q_step = jit(self.q_step, device=self.device)
        jit_policy_step = jit(self.policy_step, device=self.device)
        jit_target_step = jit(self.target_step, device=self.device)
        
        step = 0 #self.q_opt_state[0][0]
        
        q_loss = 0.0
        p_loss = 0.0
        
        
        config = dict(batch_size=self.batch_size, 
                      policy_update_interval=self.policy_update_interval,
                      initial_alpha=self.initial_alpha,
                      tau=self.tau,
                      gamma=self.gamma,
                      q_lr=self.q_lr,
                      p_lr=self.p_lr,
                      total_steps=self.total_steps,
                      warmup_steps = self.warmup_steps
                      ) 
        config.update(self.config)
        
        reporter = Reporter(
            use_tensorboard=config['report_to_tensorboard'],  
            use_wandb=config['report_to_wandb'],       
            tensorboard_dir="runs/",
            wandb_project='sac-my-flax-arena-4',
            wandb_config=config,
        )
        
        os.makedirs(os.path.join('checkpoints', reporter.run_name), exist_ok=True)
        
        logger.debug("Waiting for worker to start consuming queued agents")
        
        for _ in range(self.agent_queue.maxsize+2):  
            self.agent_queue.put((reporter.run_name, self.agent_params, self.qf1_params, self.qf2_params, step, (self.q_opt_state, self.p_opt_state, self.alpha_opt_state)), block=self.synchroneous)
        
        logger.debug("Starting trainer loop")
        
        while self.running:
            batch = self.buffer_thread.sample(self.batch_size)
            observations, next_observations, actions, rewards, dones = [jax.device_put(data, self.device) for data in batch]
            
            mean_reward = rewards.mean()
            
            inference_key0, inference_key1, self.rng_key = jax.random.split(self.rng_key, 3)
            
            
        
            q_loss, self.qf1_params, self.qf2_params, self.q_opt_state = jit_q_step(self.q_opt_state,
                                                                                    observations, next_observations, actions, rewards, dones, 
                                                                                    self.qf1_params, self.qf2_params, 
                                                                                    self.qf_target1_params, self.qf_target2_params, 
                                                                                    self.agent_params,
                                                                                    inference_key0, self.alpha)
            
            if self.policy_update_interval > 0 and step % self.policy_update_interval == 0:
                p_loss, a_loss, self.agent_params, self.alpha, self.log_alpha, self.p_opt_state, self.alpha_opt_state = jit_policy_step(self.p_opt_state,
                                                                                                                                self.alpha_opt_state,
                                                                                                                                observations, 
                                                                                                                                self.agent_params, 
                                                                                                                                self.qf1_params, self.qf2_params, 
                                                                                                                                inference_key1, 
                                                                                                                                self.alpha, self.log_alpha)
            
            self.qf_target1_params, self.qf_target2_params = jit_target_step(self.qf_target1_params, self.qf_target2_params, 
                                                                                self.qf1_params, self.qf2_params)
            
            episode_reward = jax.numpy.array(self.worker_thread.episode_reward).mean().item()
            
                        
            if step % 25 == 0 or step>=self.total_steps:
                current_p_lr = self.p_lr_schedule(step)
                current_q_lr = self.q_lr_schedule(step)
                
                data = {
                    'q-loss': float(q_loss), 
                    'p-loss': float(p_loss), 
                    'a-loss': float(a_loss),
                    'alpha': float(self.alpha),
                    'reward-buffer-mean': float(mean_reward), 
                    'reward-episode': float(episode_reward), 
                    'p_lr': float(current_p_lr),
                    'q_lr': float(current_q_lr),
                    'env-steps': self.worker_thread.step
                }
                
                validator_str = ''

                if self.worker_thread.validant_last_reward is not None:
                    data['validator-last-reward'] = float(self.worker_thread.validant_last_reward)
                    data['validator-best-reward'] = float(self.worker_thread.validant_best_reward)
                    validator_str =f', \tv-last: {self.worker_thread.validant_last_reward:1.2f},\tv-best: {self.worker_thread.validant_best_reward:1.2f}'
                ratio = float(step)/self.worker_thread.step if self.worker_thread.step else 0.0
                
                print(f'{step:05d}, env step: {self.worker_thread.step}, ratio: {ratio:4.2f} \t\t'
                        f'q-loss: {q_loss:8.4f}, p-loss: {p_loss:8.4f}, \t'
                        f'a-loss: {a_loss:8.4f}, alpha: {self.alpha:6.4f}, \t'
                        f'mean reward: {mean_reward:8.4f}, episode_reward: {episode_reward:8.4f}, \t'
                        f'p_lr: {current_p_lr:8.5f}, q_lr: {current_q_lr:8.5f}'
                        f'{validator_str}')

                reporter.log(data, step=step)
                    
            
            if self.worker_thread.step>100 and self.synchroneous:
                logger.debug('Synchroneous mode disabled')
                self.synchroneous = False
            try:
                self.agent_queue.put((reporter.run_name, self.agent_params, self.qf1_params, self.qf2_params, step, (self.q_opt_state, self.p_opt_state, self.alpha_opt_state)), block=self.synchroneous)
            except Full:
                pass
            
            if step % 250 == 0 or step>=self.total_steps:
                self.save_state(os.path.join('checkpoints', reporter.run_name, 'last'))
            
            if step>=self.total_steps:
                break
            step += 1
        logger.info(f"MultiSAC-MJX {self.name} finished!")
 
    
       
def main():
    agent_queue = Queue()
    env = HumanoidEnv()
    agent = ActorSimple(jax.random.key(42), 
                env.action_space_shape[0], 
                env.ctrlrange_high, 
                env.ctrlrange_low,
                256, 256)
    
    bt = BufferThread(1e4, 32, env.observation_space_shape, env.action_space_shape)
    wt = TrainingThread(buffer_thread=bt, agent_queue=agent_queue, batch_size=128, agent=agent)
    wt.start()
    wt.join()
        
if __name__ == "__main__":
    main()