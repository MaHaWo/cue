import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import pickle

# Emulator model to predict spectra based on input parameters
class EmulatorModel(nn.Module):
    hidden_layers: list
    output_dim: int

    def setup(self):
        self.layers = [nn.Dense(features=h) for h in self.hidden_layers]
        self.output_layer = nn.Dense(features=self.output_dim)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.output_layer(x)

# Load the emulator parameters from the given file
with open("/mnt/data/jax_rewrite/speculator_cont_new.pkl", "rb") as f:
    emulator_params = pickle.load(f)

hidden_layers = emulator_params['hidden_layers']
output_dim = emulator_params['output_dim']

# Instantiate the emulator model
emulator_model = EmulatorModel(hidden_layers=hidden_layers, output_dim=output_dim)

# Example state setup with dummy data
#x_dummy = jnp.array([[21.5, 14.85, 6.45, 3.15, 4.55, 0.7, 0.85, 49.58, 10**2.5, -0.85, -0.134, -0.134]])
#state = train_state.TrainState.create(
#    apply_fn=emulator_model.apply,
#    params=emulator_model.init(jax.random.PRNGKey(0), x_dummy),
#    tx=optax.adam(learning_rate=0.001)
#)

# Prediction function
def predict_emulator(params, input_data):
    return emulator_model.apply(params, input_data)

# Example prediction
#emulator_prediction = predict_emulator(state.params, x_dummy)
