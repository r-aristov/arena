import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.scipy.special import logsumexp

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class ActorSimple_skip(nn.Module):
    action_space_size: int
    action_space_high: float
    action_space_low: float
    actor_fc1_size: int
    actor_fc2_size: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.actor_fc1_size, name='fc1')(obs)
        x = nn.relu(x)
        x = jnp.concatenate((obs, x), axis=-1)
        x = nn.Dense(self.actor_fc2_size, name='fc2')(x)
        x = nn.relu(x)
        x = jnp.concatenate((obs, x), axis=-1)
        
        mean = nn.Dense(self.action_space_size, name='mean')(x)
        log_std = nn.Dense(self.action_space_size, name='logstd')(x)
        log_std = jnp.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, params, x, rng_key):
        mean, log_std = self.apply({'params': params}, x)
        
        std = jnp.exp(log_std)
        
        normal = jax.random.normal(rng_key, shape=mean.shape)
        x_t = mean + std * normal
        y_t = jnp.tanh(x_t)

        action_scale = (self.action_space_high - self.action_space_low) / 2.0
        action_bias = (self.action_space_high + self.action_space_low) / 2.0

        action = y_t * action_scale + action_bias

        log_prob = -0.5 * (((x_t - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
        log_prob -= jnp.log(action_scale * (1 - y_t ** 2) + 1e-6)
        log_prob = log_prob.sum(axis=-1, keepdims=True) 

        mean = jnp.tanh(mean) * action_scale + action_bias

        return action, log_prob, mean
    

if __name__ == "__main__":
    observation_size = 73  
    action_size = 8      
    batch_size = 7
    
    agent_rng_key, init_key = jax.random.split(jax.random.PRNGKey(0))

    model = ActorSimple_skip(action_space_size=action_size, action_space_high=1.0, 
                        action_space_low=-1.0, hidden1_size=256, hidden2_size=256)
    
    # Dummy input for shape inference
    dummy_x = jnp.ones((batch_size, observation_size))  
    
    params = model.init(init_key, dummy_x)['params']
    mean, logstd = model.apply({'params': params}, dummy_x)
    print('Mean shape:', mean.shape)
    print('Logstd shape:', logstd.shape)
    

    agent_rng_key, key = jax.random.split(agent_rng_key, 2)
    action, log_prob, mean = model.get_action(params, dummy_x, key)

    print("Action shape:", action.shape)
    print("Log Probability shape:", log_prob.shape)
    print("Scaled mean shape:", mean.shape)