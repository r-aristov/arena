from collections import deque
import time
import jax
from jax import vmap
import mujoco
from mujoco import mjx
from mujoco.mjx import Data

import sys
from jax import numpy as jp
import threading
from queue import Queue

    
class BattleArena:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("models/arena.xml") #_all_collisions
        
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        self.model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        self.model.opt.iterations = 1 
        self.model.opt.ls_iterations = 4
        
        self.q_function = None
        self.q_params = None

        self.data = mujoco.MjData(self.model)
        self.mjx_model = mjx.put_model(self.model)
        
        self.frame_skip = 5
        
        self.r_arena = 5
        self.placement_radius = 1.3
        self.placement_noise_scale = 0.02
        self.placement_z = 0.9
        self.placement_z_noise_std = 0.1
        self.velocity_noise_scale = 0.1
        
        
        self.agent_count = 2
        self.observation_space_shape = (72+1, )
        self.action_space_shape = (8,)
        
        self.ctrlrange_high = 1.0
        self.ctrlrange_low = -1.0
        
        
        self.max_steps = 1000
        self.com_buffer_size = 2
        

    def reset(self, rng: jax.Array,):
        data = mjx.make_data(self.mjx_model)
        rng0, rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8 = jax.random.split(rng, 9)

        phi = jax.random.uniform(rng1, minval=0.0, maxval=2*jp.pi)
        
        pos0 = jp.array((jp.cos(phi), jp.sin(phi)))*self.placement_radius
        pos1 = -pos0 + jp.clip(self.placement_noise_scale*jax.random.normal(rng2, (2, )), -0.05, 0.05) 
        pos0 += jp.clip(self.placement_noise_scale*jax.random.normal(rng3, (2, )), -0.05, 0.05)
        
        z_noize = self.placement_z_noise_std * jp.clip(jax.random.normal(rng2, (1,)), -0.3, 0.3)
        z_pos0 = self.placement_z + z_noize 
        z_pos1 = self.placement_z + z_noize      
        
        quat0 = data.qpos[3:7]  
        quat1 = data.qpos[18:22]  

        qpos = jp.concatenate([
            pos0, z_pos0, quat0, data.qpos[7:15],
            pos1, z_pos1, quat1, data.qpos[22:]
        ])
        
        low, hi = -self.velocity_noise_scale, self.velocity_noise_scale

        qvel = data.qvel + jax.random.uniform(rng5, (self.model.nv,), minval=low, maxval=hi)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)
        
        role = jax.random.randint(rng4, (2,), minval=0, maxval=2) # remnants of the old experiment, not used, for compatibility only
        obs = self.get_obs(data, role)
        
        com0_buffer = jax.numpy.tile(data.subtree_com[1], (self.com_buffer_size, 1))
        com1_buffer = jax.numpy.tile(data.subtree_com[14], (self.com_buffer_size, 1))
        
        return dict(data=data,
                    step=0,
                    obs=obs,
                    rng=rng0, 
                    com=(data.subtree_com[1], data.subtree_com[14]),
                    role=role,
                    com_buffer=(com0_buffer, com1_buffer),
                    max_d_0=jp.zeros(1)[0],
                    max_d_1=jp.zeros(1)[0],
                    had_contact=jp.zeros(1)[0],
                    dd0=jp.zeros(1)[0],
                    dd1=jp.zeros(1)[0],
                    )
        
    
    def get_obs(self, data: Data, role):
        return jp.concatenate((jp.expand_dims(role[0], 0),
                                data.qpos[:15],
                                data.qvel[:14], 
                                data.qfrc_actuator[:14],

                                data.qpos[15:],
                                data.qvel[14:]
                                )), \
            jp.concatenate((jp.expand_dims(role[1], 0),
                            data.qpos[15:],
                            data.qvel[14:],
                            data.qfrc_actuator[14:],

                            data.qpos[:15],
                            data.qvel[:14], 
                            )) 
        
    
    def compute_reward(self, old_data, new_data, had_contact0, had_contact, dd0, dd1):
        fighter0_com = new_data.subtree_com[1]
        fighter1_com = new_data.subtree_com[14]
        fighter0_com_last = old_data.subtree_com[1]
        fighter1_com_last = old_data.subtree_com[14]

        r_vec0 = jp.sqrt(jp.sum(jp.square(fighter0_com)))
        r_vec1 = jp.sqrt(jp.sum(jp.square(fighter1_com)))
        
        
        
        dir_vec01 = fighter1_com-fighter0_com_last
        dir_vec_len = jp.sqrt(jp.sum(jp.square(dir_vec01)))
        dir_vec01 = jp.where(dir_vec_len > 0, dir_vec01 / dir_vec_len, dir_vec01*0.0)
        
        dir_vec10 = fighter0_com-fighter1_com_last
        dir_vec_len = jp.sqrt(jp.sum(jp.square(dir_vec10)))
        dir_vec10 = jp.where(dir_vec_len > 0, dir_vec10 / dir_vec_len, dir_vec10*0.0)
        
        
        movement_vec0 = fighter0_com - fighter0_com_last
        movement_projection0 = jp.clip(jp.dot(movement_vec0, dir_vec01), -0.5, 0.5)
        
        movement_vec1 = fighter1_com - fighter1_com_last
        movement_projection1 = jp.clip(jp.dot(movement_vec1, dir_vec10), -0.5, 0.5)
        

        pre_fell_to_the_hell0 = jp.where(fighter0_com[3] < 0.0 , 1.0, 0.0)
        pre_fell_to_the_hell1 = jp.where(fighter1_com[3] < 0.0 , 1.0, 0.0)
        
    
        k3 = 10.0
        k4 = 3.0
        
        dd_reward0 = jp.clip(jp.where(r_vec0<r_vec1, dd1, 0.0), -0.5, 0.5)
        dd_reward1 = jp.clip(jp.where(r_vec1<r_vec0, dd0, 0.0), -0.5, 0.5)
        
        
        a0_reward = k4*movement_projection0 + k3*dd_reward0 
        a1_reward = k4*movement_projection1 + k3*dd_reward1 
        a0_reward = jp.where(had_contact0+had_contact==1.0, 0.5, a0_reward)
        a1_reward = jp.where(had_contact0+had_contact==1.0, 0.5, a1_reward)
        
        
        a0_reward = jp.where(pre_fell_to_the_hell0+pre_fell_to_the_hell1>0, 0.0, a0_reward)
        a1_reward = jp.where(pre_fell_to_the_hell0+pre_fell_to_the_hell1>0, 0.0, a1_reward)
        
        k0 = -0.2
        k1 = 0.2
        a0_reward = jp.where(pre_fell_to_the_hell0 > 0, k0, jp.where(pre_fell_to_the_hell1 > 0, k1, a0_reward))
        a1_reward = jp.where(pre_fell_to_the_hell1 > 0, k0, jp.where(pre_fell_to_the_hell0 > 0, k1, a1_reward))
        
        return a0_reward, a1_reward
    

    def validation_reward(self, fighter0_com, fighter1_com):
        fell_to_the_hell0 = jp.where(fighter0_com[3] < 0.0 , 1.0, 0.0)
        fell_to_the_hell1 = jp.where(fighter1_com[3] < 0.0 , 1.0, 0.0)
        a0_reward = jp.where(fell_to_the_hell0<fell_to_the_hell1, 1.0, jp.where(fell_to_the_hell0>fell_to_the_hell1, -1.0, 0.0))
        a1_reward = jp.where(fell_to_the_hell1<fell_to_the_hell0, 1.0, jp.where(fell_to_the_hell1>fell_to_the_hell0, -1.0, 0.0))
        return a0_reward, a1_reward
    
    
    def step(self, state0, control):
        data, _ = jax.lax.scan(lambda d0, _: (mjx.step(self.mjx_model, d0.replace(ctrl=control)), None), state0['data'], (), self.frame_skip)
        new_step = state0['step'] + 1
        steps_limit_reached = jp.where(new_step >= self.max_steps, 1.0, 0.0)
          
        fighter0_com = data.subtree_com[1]
        fighter1_com = data.subtree_com[14]
        
        com0_buffer, com1_buffer = state0['com_buffer']
        com0_buffer = jp.concat((com0_buffer[1:], jp.expand_dims(fighter0_com, 0)))
        com1_buffer = jp.concat((com1_buffer[1:], jp.expand_dims(fighter1_com, 0)))
        
        f0f1_distance = jp.sqrt(jp.sum(jp.square(fighter1_com[:2]-fighter0_com[:2])))
        had_contact = jp.where(f0f1_distance<1.0, 1.0, state0['had_contact'])   
        
        distances0 = jp.sum(jp.square(com0_buffer[:, :2]), axis=-1)
        max_d_0 = jp.where(had_contact, jp.maximum(state0['max_d_0'], jp.mean(jp.sqrt(distances0))), 0.0)
        
        distances1 = jp.sum(jp.square(com1_buffer[:, :2]), axis=-1)
        max_d_1 = jp.where(had_contact, jp.maximum(state0['max_d_1'], jp.mean(jp.sqrt(distances1))), 0.0)
        
        dd0 = max_d_0 - state0['max_d_0']
        dd1 = max_d_1 - state0['max_d_1']
                
        reward0, reward1 = self.compute_reward(state0['data'], data, state0['had_contact'], had_contact, dd0, dd1)
        validation_r0, validation_r1 = self.validation_reward(fighter0_com, fighter1_com)
        
        # terminal rewards 
            
        fell_to_hell_limit = -30.0
        pre_fell_to_hell_limit = -1.5
        fell_to_the_hell0 = jp.where(fighter0_com[3] < fell_to_hell_limit , 1.0, 0.0)
        fell_to_the_hell1 = jp.where(fighter1_com[3] < fell_to_hell_limit , 1.0, 0.0)
        pre_fell_to_the_hell0 = jp.where(fighter0_com[3] < pre_fell_to_hell_limit , 1.0, 0.0)
        pre_fell_to_the_hell1 = jp.where(fighter1_com[3] < pre_fell_to_hell_limit , 1.0, 0.0)
        
        both_fell = fell_to_the_hell0*pre_fell_to_the_hell1 + fell_to_the_hell1*pre_fell_to_the_hell0
        both_survived = (1-pre_fell_to_the_hell0)*(1-pre_fell_to_the_hell1)
        
        both_fell_reward = 0.0
        both_survived_reward = -10.0
        win_reward = 0.0
        loose_reward = 0.0
        
        done = jp.where(steps_limit_reached + fell_to_the_hell0 + fell_to_the_hell1, 
                        1.0, 
                        0.0)
        
        finished_reward_0 = jp.where(both_fell>0, both_fell_reward, 
                                     jp.where(both_survived>0, both_survived_reward,
                                               jp.where(pre_fell_to_the_hell0>0, loose_reward, win_reward)))
        finished_reward_1 = jp.where(both_fell>0, both_fell_reward, 
                                     jp.where(both_survived>0, both_survived_reward,
                                               jp.where(pre_fell_to_the_hell1>0, loose_reward, win_reward)))
        
        reward0 = jp.where(done > 0, finished_reward_0, reward0)
        reward1 = jp.where(done > 0, finished_reward_1, reward1)
        
        observations = self.get_obs(data, state0['role'])
        
        new_state = dict(data=data, 
                         step=new_step, 
                         obs=observations, 
                         rng=state0['rng'], 
                         com=(fighter0_com, fighter1_com), 
                         role=state0['role'],
                         com_buffer=(com0_buffer, com1_buffer),
                         max_d_0=max_d_0,
                         max_d_1=max_d_1,
                         had_contact=had_contact,
                         dd0 = dd0,
                         dd1 = dd1) 
        next_state = jax.lax.cond(done, self.reset, lambda _ : new_state, new_state['rng'])
        
        return next_state, new_state["obs"], (jax.numpy.array([reward0]), jax.numpy.array([reward1])), jax.numpy.array([done]), (validation_r0, validation_r1)



def parse_args():
    parser = argparse.ArgumentParser(description='Battle Arena for RL agents')
    
    # Agent paths
    parser.add_argument('--agent0', type=str, default=None,
                        help='Path to agent 0 (default: agent.flax in root or eternal-firefly-57 if no agent.flax found)')
    parser.add_argument('--agent1', type=str, default='legacy-agents/eternal-firefly-57/agent.flax',
                        help='Path to agent 1 (default: eternal-firefly-57)')
    
    # Normalization values
    parser.add_argument('--norm0', type=str, default='obs-norm/obs_norm_legacy.npz',
                        help='Path to normalization values for agent 0')
    parser.add_argument('--norm1', type=str, default='obs-norm/obs_norm_legacy.npz',
                        help='Path to normalization values for agent 1')
    
    # Saving options
    parser.add_argument('--save-mode', type=str, choices=['none', 'images', 'positions', 'both'],
                        default='none', help='What data to save')
    
    args = parser.parse_args()
    
    # If agent0 is not specified, try to use agent.flax from root, else use eternal-firefly-57
    if args.agent0 is None:
        if os.path.exists('agent.flax'):
            args.agent0 = 'agent.flax'
        else:
            args.agent0 = 'legacy-agents/eternal-firefly-57/agent.flax'
    
    return args


def get_agent_display_name(agent_path):
    dirname = os.path.dirname(agent_path)
    if dirname == '':
        return f"./{os.path.basename(agent_path)}"
    return dirname


def finish(image_queue, image_saver_thread):
    if image_queue is not None:
        image_queue.put((None, None, None, None))
        print("Waiting frames to be saved as images...")
        image_saver_thread.join()
    sys.exit(0)


def save_images(queue):
    while True:
        frame, pixels, xpos, xmat = queue.get()
        if frame is None:
            break
        if xpos is not None and xmat is not None:
            np.savez(f"outputs/pos-data/{frame:05d}.npz", geom_xpos=xpos, geom_xmat=xmat)
        if pixels is not None:
            plt.imsave(f"outputs/images/{frame:05d}.png", pixels)
        
        
def normalize_obs(obs, obs_rms_mean, obs_rms_var):
    return jax.numpy.clip((obs - obs_rms_mean) / jax.numpy.sqrt(obs_rms_var + 1e-8), -10.0, 10.0)
            
            
def main():
    args = parse_args()
    
    # Print starting parameters
    print("Starting Battle Arena with parameters:")
    print(f"Agent 0: {args.agent0}")
    print(f"Agent 1: {args.agent1}")
    print(f"Normalization 0: {args.norm0}")
    print(f"Normalization 1: {args.norm1}")
    print(f"Save mode: {args.save_mode}")

    # check if everything works ok with batches
    ba = BattleArena()
    batch_size = 64
    key = jax.random.split(jax.random.key(42), batch_size)
    batched_state = vmap(ba.reset)(key)
    print('batch qpos shape', batched_state['data'].qpos.shape)

    agent1 = ActorSimple_skip(ba.action_space_shape[0], 
                    ba.ctrlrange_high, ba.ctrlrange_low,
                    512, 512) 
    agent0 = ActorSimple_skip(ba.action_space_shape[0], 
                    ba.ctrlrange_high, ba.ctrlrange_low,
                    512, 512) 
    
    # Load agents
    with open(args.agent0, 'rb') as f:
        agent_0_bytes = f.read()
    with open(args.agent1, 'rb') as f:
        agent_1_bytes = f.read()
    
    # Load normalization values
    obs_norm_values = np.load(args.norm0)
    obs_rms_mean0 = jax.numpy.array(obs_norm_values['obs_mean'])
    obs_rms_var0 = jax.numpy.array(obs_norm_values['obs_var'])
    
    obs_norm_values = np.load(args.norm1)
    obs_rms_mean1 = jax.numpy.array(obs_norm_values['obs_mean'])
    obs_rms_var1 = jax.numpy.array(obs_norm_values['obs_var'])
    
    # Initialize image saving only if needed
    image_queue = None
    image_saver_thread = None
    if args.save_mode != 'none':
        os.makedirs('outputs', exist_ok=True)
        os.makedirs(os.path.join('outputs', 'images'), exist_ok=True)
        os.makedirs(os.path.join('outputs', 'pos-data'), exist_ok=True)
        image_queue = Queue()
        image_saver_thread = threading.Thread(target=save_images, args=(image_queue,))
        image_saver_thread.start()
    
    inference_key, env_key, init_key = jax.random.split(jax.random.key(42), 3)
    
    jit_reset = jax.jit(ba.reset)
    
    state = jit_reset(env_key)
                        
    agent_0_params = serialization.from_bytes(agent0.init(init_key, batched_state['obs'][0])['params'], agent_0_bytes)
    agent_1_params = serialization.from_bytes(agent1.init(init_key,  batched_state['obs'][1])['params'], agent_1_bytes)
    
    width = 1280
    height = 720
    
    ba.model.vis.global_.offwidth = width
    ba.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(ba.model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera_id = mujoco.mj_name2id(ba.model, mujoco.mjtObj.mjOBJ_CAMERA, 'free_cam', )
    camera.fixedcamid = camera_id
    camera.type = 2
    camera.trackbodyid = 0
    
    jit_step = jax.jit(ba.step)

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"{args.agent0} vs {args.agent1}")
    clock = pygame.time.Clock()
    keymap = {pygame.K_ESCAPE: 0.0, pygame.K_SPACE: 0.0}

    rewards0 = deque([0.0]*1000, maxlen=1000)
    rewards1 = deque([0.0]*1000, maxlen=1000)
    frame = 0
    
    while True:
        wheel = 0
        for event in pygame.event.get():
                if event.type == QUIT:
                    finish(image_queue, image_saver_thread)
                if event.type == pygame.KEYDOWN:
                    keymap[event.key] = 1.0
                if event.type == pygame.KEYUP:
                    keymap[event.key] = 0.0
                if event.type == pygame.MOUSEWHEEL:
                    wheel = event.y

        if keymap[pygame.K_ESCAPE] > 0.0:
            finish(image_queue, image_saver_thread)
        
        if keymap[pygame.K_SPACE] > 0.0:
            state = jit_reset(jax.random.key(time.time_ns()))
            
        mjx.get_data_into(ba.data, ba.model, state['data'])
        mujoco.mj_forward(ba.model, ba.data)
        renderer.update_scene(ba.data, camera=camera)
        pixels = renderer.render()
        
        if image_queue is not None:
            if args.save_mode == 'images' or args.save_mode == 'both':
                pixels_to_save = pixels
            else:
                pixels_to_save = None
                
            if args.save_mode == 'positions' or args.save_mode == 'both':
                xpos_to_save = ba.data.xpos
                xmat_to_save = ba.data.xmat
            else:
                xpos_to_save = None
                xmat_to_save = None
                
            image_queue.put((frame, pixels_to_save, xpos_to_save, xmat_to_save))
        
        pixels = np.swapaxes(pixels, 0, 1)
        pix_surf = pygame.surfarray.make_surface(pixels)
    
        screen.blit(pix_surf, (0, 0))

        scale = 40
        r = 0.25
        com0, com1 = state['com']
        com0 = com0*scale
        com1 = com1*scale
        cx = 1080
        cy = 600
        arena_hwidth = 2.5
        arena_hheight = 2.5
        
        pygame.draw.rect(screen, 
                         (0, 255,0),
                         (cx-scale*arena_hwidth, cy-scale*arena_hheight, scale*2*arena_hwidth, scale*2*arena_hheight),
                         2)
        pygame.draw.circle(screen, 
                           (255,0,0),
                           (int(cx+com0[0]), int(cy-com0[1])),
                            r*scale)
        pygame.draw.circle(screen, 
                           (0,0,255),
                           (int(cx+com1[0]), int(cy-com1[1])),
                           r*scale)
        if state['had_contact'] > 0:
            pygame.draw.circle(screen, 
                           (255,0,0),
                           (int(cx), int(cy)),
                           int(state['max_d_0']*scale),
                           1)
            
            pygame.draw.circle(screen, 
                           (0,0,255),
                           (int(cx), int(cy)),
                           int(state['max_d_1']*scale),
                           1)
            
        y_coord = 150 - 70*np.array(rewards0) 
        x_coord = 140 + np.arange(1000)
        reward_points = np.stack([x_coord, y_coord])
        pygame.draw.lines(screen,
                          (255, 0, 0),
                          False,
                          reward_points.T,
                          2
                          ) 
        
        y_coord = 150 - 70*np.array(rewards1) 
        reward_points = np.stack([x_coord, y_coord])
        pygame.draw.lines(screen,
                          (0, 0, 255),
                          False,
                          reward_points.T,
                          2)
            
        
        key0, key1, inference_key = jax.random.split(inference_key, 3)
        batched_obs = jp.expand_dims(state['obs'][0], 0)
        
        action, _, _ = agent0.get_action(agent_0_params, normalize_obs(batched_obs, obs_rms_mean0, obs_rms_var0), key0)
        action1, _, _ = agent1.get_action(agent_1_params, normalize_obs(state['obs'][1], obs_rms_mean1, obs_rms_var1), key1)  
   
        actions = jax.numpy.concat([
            jp.squeeze(action),
            action1
        ])
    
        state, _, rewards, _, validation_rewards = jit_step(state, actions)
        
        pygame.display.flip()
        rewards0.append(rewards[0].item())
        rewards1.append(rewards[1].item())
        
        print(state['step'], validation_rewards[0], validation_rewards[1], rewards[0], rewards[1], clock.get_fps())
       
        clock.tick(50)
        frame += 1            
        
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from flax import serialization
    import pygame
    from pygame.locals import QUIT
    from flax import serialization
    from agent import ActorSimple_skip
    import numpy as np
    
    import argparse
    import os
    import sys
    import time
    import threading
    from queue import Queue
    from collections import deque
    main()

