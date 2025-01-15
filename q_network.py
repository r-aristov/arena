import jax
import jax.numpy as jnp
from flax import linen as nn
    
class SoftQNetwork_skip(nn.Module):
    hidden_size1: int = 512  
    hidden_size2: int = 512  

    @nn.compact
    def __call__(self, x, a):
        concatenated_input = jnp.concatenate([x, a], axis=-1)
        
        x = nn.Dense(self.hidden_size1)(concatenated_input)
        x = nn.relu(x)
        x = jnp.concatenate([x, concatenated_input], axis=-1)

        x = nn.Dense(self.hidden_size2)(x)
        x = nn.relu(x)
        x = jnp.concatenate([x, concatenated_input], axis=-1)

        x = nn.Dense(1)(x)
        return x
    

if __name__ == "__main__":

    observation_size = 73  
    action_size = 8      
    batch_size = 7
    
    model = SoftQNetwork_skip(hidden1_size=256, hidden2_size=256)

    rng_key = jax.random.PRNGKey(0)
    
    # Dummy inputs for shape inference
    dummy_obs = jnp.ones((batch_size, observation_size))  
    dummy_act = jnp.ones((batch_size, action_size))     
    
    params = model.init(rng_key, dummy_obs, dummy_act)['params']
    
    q_values = model.apply({'params': params}, dummy_obs, dummy_act)
    print("Q Values:", q_values.shape)