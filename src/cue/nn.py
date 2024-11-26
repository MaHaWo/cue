import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import pickle

# Load the PCA and Speculator models using new Flax-based approach
class PCAModel(nn.Module):
    num_components: int

    def setup(self):
        self.linear = nn.Dense(features=self.num_components)

    def __call__(self, x):
        return self.linear(x)


class SpeculatorModel(nn.Module):
    hidden_layers: list
    output_dim: int

    def setup(self):
        self.layers = [nn.Dense(features=h) for h in self.hidden_layers]
        self.output_layer = nn.Dense(features=self.output_dim)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.output_layer(x)


# Load PCA parameters
with open("/mnt/data/jax_rewrite/pca_cont_new.pkl", "rb") as f:
    pca_params = pickle.load(f)

num_pca_components = pca_params['num_components']

# Instantiate PCA model
pca_model = PCAModel(num_components=num_pca_components)

# Load Speculator parameters
with open("/mnt/data/jax_rewrite/speculator_cont_new.pkl", "rb") as f:
    speculator_params = pickle.load(f)

hidden_layers = speculator_params['hidden_layers']
output_dim = speculator_params['output_dim']

# Instantiate Speculator model
speculator_model = SpeculatorModel(hidden_layers=hidden_layers, output_dim=output_dim)


# Example state setup with dummy data (assuming optax optimizer)
#x_dummy = jnp.array([[21.5, 14.85, 6.45, 3.15, 4.55, 0.7, 0.85, 49.58, 10**2.5, -0.85, -0.134, -0.134]])
#state = train_state.TrainState.create(
#    apply_fn=pca_model.apply,
#    params=pca_model.init(jax.random.PRNGKey(0), x_dummy),
#    tx=optax.adam(learning_rate=0.001)
#)

# Prediction using the new PCA model
def predict_pca(params, input_data):
    return pca_model.apply(params, input_data)

#pca_prediction = predict_pca(state.params, x_dummy)

# Similarly, instantiate Speculator state and make predictions
