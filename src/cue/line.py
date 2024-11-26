import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import pickle

# Line model for predicting spectra using Flax
class LineModel(nn.Module):
    hidden_layers: list
    output_dim: int

    def setup(self):
        self.layers = [nn.Dense(features=h) for h in self.hidden_layers]
        self.output_layer = nn.Dense(features=self.output_dim)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.output_layer(x)

# Load the line parameters from the given file
with open("data/speculator_line_new_C4.pkl", "rb") as f:
    line_params = pickle.load(f)

hidden_layers = line_params['hidden_layers']
output_dim = line_params['output_dim']

# Instantiate the line model
line_model = LineModel(hidden_layers=hidden_layers, output_dim=output_dim)

# Train state setup
state = train_state.TrainState.create(
    apply_fn=line_model.apply,
    params=line_model.init(jax.random.PRNGKey(0), jnp.ones((1, len(hidden_layers)))),
    tx=optax.adam(learning_rate=0.001)
)

# Prediction function for the line model
def predict_line(input_data):
    class LinePredictor:
        def __init__(self, params):
            self.params = params

        def nn_predict(self):
            return line_model.apply(self.params, input_data)

    return LinePredictor(state.params)
