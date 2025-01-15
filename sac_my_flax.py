import logging
from worker import WorkerThread
from buffer import BufferThread
from trainer import TrainingThread
from arena import BattleArena
from agent import ActorSimple_skip

import jax
from queue import Queue
from flax import serialization

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.basicConfig(
    format='%(asctime)s [%(threadName)s] %(levelname)s: %(message)s',
    level=logging.WARNING,
    force=True
)

def load_agent(path, template):
    with open(path, 'rb') as checkpoint:
            return serialization.from_bytes(template, checkpoint.read())
        
devices = jax.devices()
logger.info(f'Available devices: {devices}')
worker_device = devices[0]
trainer_device = devices[1] if len(devices) > 1 else worker_device


ref_agents_paths = ['legacy-agents/twilight-durian-52/agent.flax',
                   'legacy-agents/super-bird-53/agent.flax',           
                   'legacy-agents/glamorous-dust-67/agent.flax',                     
                   'legacy-agents/eternal-firefly-57/agent.flax',
                    ]
ref_agents_paths=ref_agents_paths*4

config = dict(
    seed = 42,
    worker_device=worker_device,
    trainer_device=trainer_device,
    ref_agents_count=len(ref_agents_paths),

    trainer_batch_size=4096*2,
    worker_batch_size=1024*4,
    buffer_size=4*4.096e6,
    
    random_steps_count=2000,
    last_agents_count=16,
    self_play_lag = 2501,
    # trainer metaparameters
    initial_alpha = 1.0,
    autotune_alpha = True,
    
    tau = 0.005,
    gamma = 0.985,
    q_lr = 0.001,
    p_lr = 0.001,
    
    total_steps=2_000_000,
    warmup_steps=40_000,
    
    report_to_tensorboard=False,
    report_to_wandb=True
    )

_, worker_key, trainer_key, _, init_key = jax.random.split(jax.random.key(config['seed']), 5)

env = BattleArena()

agent_skip = ActorSimple_skip(env.action_space_shape[0], 
                    env.ctrlrange_high, env.ctrlrange_low,
                    512, 512) 

validation_agent_paths = [
            'legacy-agents/playful-capybara-68/agent.flax', # 10.85  
            'legacy-agents/eternal-firefly-57/agent.flax', # 9.86     
            'legacy-agents/super-bird-53/agent.flax', # 9.81
            'legacy-agents/toasty-pine-70/agent.flax', # 3.22
            'legacy-agents/glamorous-dust-67/agent.flax', # 0.30
            'legacy-agents/playful-wildflower-34/agent.flax', # -0.82
            
            'legacy-agents/twilight-durian-52/agent.flax', # -1.63
            'legacy-agents/polished-voice-50/agent.flax', # -3.22
            
            ]
                   
state0 = env.reset(jax.random.key(0))

obs = state0['obs'][0]
observations_count = jax.numpy.prod(jax.numpy.array(obs.shape)).item()
agent_params = agent_skip.init(init_key, jax.numpy.ones((1, observations_count)))['params']
 

ref_agent_params = []
opponent_funcs = []
dummy_params = [agent_skip.init(worker_key, jax.numpy.ones((1, observations_count)))['params']]*len(ref_agents_paths)

if len(ref_agents_paths) > 0:
    for agent_path, dummy_param in zip(ref_agents_paths, dummy_params):
        with open(agent_path, 'rb') as f:    
            logger.info(f'Loaded ref agent {agent_path}')
            ref_agent_params.append(serialization.from_bytes(dummy_param, f.read()))

dummy_params = [agent_skip.init(worker_key, jax.numpy.ones((1, observations_count)))['params']]*len(validation_agent_paths)
validation_agent_params = []   
for agent_path, dummy_param in zip(validation_agent_paths, dummy_params):
    with open(agent_path, 'rb') as f:    
        logger.info(f'Loaded validation agent {agent_path}')
        validation_agent_params.append(serialization.from_bytes(dummy_param, f.read()))
        

sims_per_validation_agent = 50
validation_agents_count = len(validation_agent_params)
validation_batch_size = validation_agents_count * sims_per_validation_agent
total_batch_size = config['worker_batch_size'] + validation_batch_size
        
worker_agent_queue = Queue(8)

bt = BufferThread(config['buffer_size'], total_batch_size, env.observation_space_shape, env.action_space_shape)

wt = WorkerThread(config=config,
                  env=env, 
                  rng_key=worker_key, 
                  result_queue=bt.input_queue, 
                  agent_queue=worker_agent_queue, 
                  agent_config=dict(
                      trainee_func=agent_skip.get_action,
                      ref_agent_func=agent_skip.get_action,
                      validation_agent_func=agent_skip.get_action,
                      ref_agent_params=ref_agent_params,
                      validation_agent_params=validation_agent_params
                      ),
                  name='Worker-0')


tt = TrainingThread(config=config,
                    buffer_thread=bt,
                    agent_queue=worker_agent_queue,
                    agent=agent_skip,
                    agent_params=agent_params,
                    rng_key=trainer_key,
                    observation_space_shape=env.observation_space_shape,
                    action_space_shape=env.action_space_shape,
                    worker_thread=wt)

bt.start()
wt.start()

tt.start()

tt.join()
wt.running=False
tt.save_state('final')