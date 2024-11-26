import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import pickle

# Load PCA parameters for the continuum
with open("/mnt/data/jax_rewrite/pca_cont_new.pkl", "rb") as f:
    pca_cont_params = pickle.load(f)

num_pca_cont_components = pca_cont_params['num_components']

# Continuum PCA model class
class ContinuumPCAModel(nn.Module):
    num_components: int

    def setup(self):
        self.linear = nn.Dense(features=self.num_components)

    def __call__(self, x):
        return self.linear(x)

# Instantiate continuum PCA model
cont_pca_model = ContinuumPCAModel(num_components=num_pca_cont_components)

# Load PCA parameters for the line
with open("/mnt/data/jax_rewrite/pca_line_new_Ne4.pkl", "rb") as f:
    pca_line_params = pickle.load(f)

num_pca_line_components = pca_line_params['num_components']

# Line PCA model class
class LinePCAModel(nn.Module):
    num_components: int

    def setup(self):
        self.linear = nn.Dense(features=self.num_components)

    def __call__(self, x):
        return self.linear(x)

# Instantiate line PCA model
line_pca_model = LinePCAModel(num_components=num_pca_line_components)

# Example state setup with dummy data (assuming optax optimizer)
x_dummy = jnp.array([[21.5, 14.85, 6.45, 3.15, 4.55, 0.7, 0.85, 49.58, 10**2.5, -0.85, -0.134, -0.134]])

# Setup train state for continuum PCA model
cont_state = train_state.TrainState.create(
    apply_fn=cont_pca_model.apply,
    params=cont_pca_model.init(jax.random.PRNGKey(0), x_dummy),
    tx=optax.adam(learning_rate=0.001)
)

# Setup train state for line PCA model
line_state = train_state.TrainState.create(
    apply_fn=line_pca_model.apply,
    params=line_pca_model.init(jax.random.PRNGKey(0), x_dummy),
    tx=optax.adam(learning_rate=0.001)
)

# Prediction functions
def predict_cont_pca(params, input_data):
    return cont_pca_model.apply(params, input_data)

def predict_line_pca(params, input_data):
    return line_pca_model.apply(params, input_data)

# Make predictions
#cont_pca_prediction = predict_cont_pca(cont_state.params, x_dummy)
#line_pca_prediction = predict_line_pca(line_state.params, x_dummy)
