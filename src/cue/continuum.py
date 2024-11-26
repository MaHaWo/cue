import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import pickle

# Continuum model for predicting spectra using Flax
class ContinuumModel(nn.Module):
    hidden_layers: list
    output_dim: int

    def setup(self):
        self.layers = [nn.Dense(features=h) for h in self.hidden_layers]
        self.output_layer = nn.Dense(features=self.output_dim)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.output_layer(x)

# Load the continuum parameters from the given file
with open("data/speculator_cont_new.pkl", "rb") as f:
    continuum_params = pickle.load(f)

hidden_layers = continuum_params['hidden_layers']
output_dim = continuum_params['output_dim']

# Instantiate the continuum model
continuum_model = ContinuumModel(hidden_layers=hidden_layers, output_dim=output_dim)

# Train state setup
state = train_state.TrainState.create(
    apply_fn=continuum_model.apply,
    params=continuum_model.init(jax.random.PRNGKey(0), jnp.ones((1, len(hidden_layers)))),
    tx=optax.adam(learning_rate=0.001)
)

# Prediction function for the continuum model
def predict_continuum(input_data):
    class ContinuumPredictor:
        def __init__(self, params):
            self.params = params

        def nn_predict(self):
            return continuum_model.apply(self.params, input_data)

    return ContinuumPredictor(state.params)


