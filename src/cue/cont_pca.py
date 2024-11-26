import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import pickle

# Continuum PCA model using Flax
class ContinuumPCAModel(nn.Module):
    num_components: int

    def setup(self):
        self.linear = nn.Dense(features=self.num_components)

    def __call__(self, x):
        return self.linear(x)

# Load PCA parameters for the continuum
with open("data/pca_cont_new.pkl", "rb") as f:
    pca_cont_params = pickle.load(f)

num_pca_components = pca_cont_params['num_components']

# Instantiate continuum PCA model
cont_pca_model = ContinuumPCAModel(num_components=num_pca_components)

