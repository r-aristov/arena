import threading
from queue import Queue
import time
import numpy as np

class BufferThread(threading.Thread):
    def __init__(self, buffer_size, env_batch_size, observation_space_shape, action_space_shape, name='Buffer Thread'):
        super(BufferThread, self).__init__(name=name)
        self.daemon = True
        self.input_queue = Queue()
        self.running = False
        self.buffer = ReplayBuffer(buffer_size, env_batch_size, observation_space_shape,
                                   action_space_shape)
        self.output_count = 0
        self.total_replay_samples = 0
        

    @property
    def input_sps(self):
        return self._input_sps

    @property
    def output_sps(self):
        return self._output_sps

    def stop(self):
        self.running = False

    def run(self) -> None:
        self.running = True
        iteration = 0

        input_count = 0

        last_t = time.time()

        while self.running:
            obs, next_obs, action, reward, terminated = self.input_queue.get(block=True)
            
            self.buffer.add(obs, next_obs, action, reward, terminated)
            input_count += obs.shape[0]
            self.total_replay_samples += obs.shape[0]
            dt = time.time() - last_t
            if dt >= 1.0:
                self._input_sps = input_count / dt
                self._output_sps = self.output_count / dt
                input_count = 0
                self.output_count = 0
                last_t = time.time()

            iteration += 1

    def sample(self, batch_size):
        self.output_count += batch_size
        return self.buffer.sample(batch_size)


class ReplayBuffer:
    def __init__(self, size, env_batch_size, observation_space_shape, action_space_shape):
        self.size = int(np.ceil(size / env_batch_size)) * env_batch_size
        self.env_batch_size = env_batch_size
        self.obs_buffer = np.zeros((self.size, *observation_space_shape))
        self.next_obs_buffer = np.zeros_like(self.obs_buffer)
        self.action_buffer = np.zeros((self.size, *action_space_shape))
        self.reward_buffer = np.zeros((self.size, 1))
        self.terminated_buffer = np.zeros((self.size, 1))
        self.ptr = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, terminated):
        self.obs_buffer[self.ptr:self.ptr + self.env_batch_size] = obs
        self.next_obs_buffer[self.ptr:self.ptr + self.env_batch_size] = next_obs
        self.action_buffer[self.ptr:self.ptr + self.env_batch_size] = action
        self.reward_buffer[self.ptr:self.ptr + self.env_batch_size] = reward
        self.terminated_buffer[self.ptr:self.ptr + self.env_batch_size] = terminated
        self.ptr += self.env_batch_size

        if self.ptr >= self.size:
            if not self.full:
                print('Buffer is full')
            self.full = True
            self.ptr = 0

    def sample(self, batch_size):
        p = self.ptr if not self.full else self.size
        rng = np.random.default_rng()
        idx = rng.choice(p, batch_size)
        return self.obs_buffer[idx], self.next_obs_buffer[idx], self.action_buffer[idx], \
            self.reward_buffer[idx], \
            self.terminated_buffer[idx]