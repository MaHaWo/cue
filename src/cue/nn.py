### Speculator Neural Net
import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Tuple
import pickle


class Speculator:
    def __init__(self, n_parameters=None, wavelengths=None, pca_transform_matrix=None,
                 parameters_shift=None, parameters_scale=None, pca_shift=None, pca_scale=None,
                 log_spectrum_shift=None, log_spectrum_scale=None, n_hidden=[50, 50],
                 restore=False, restore_filename=None, optimizer=None, key=None):

        # Set up the random key
        if key is None:
            key = jax.random.PRNGKey(0)  # Default seed for reproducibility
        self.key = key
        
        # If restoring, load the model and set attributes before using them
        if restore:
            self.restore(restore_filename)

            # Set values restored from file
            n_parameters = self.n_parameters
            n_pcas = self.n_pcas
            n_wavelengths = self.n_wavelengths
            pca_transform_matrix = self.pca_transform_matrix_

        # Ensure `n_parameters` is set
        if n_parameters is None:
            raise ValueError("`n_parameters` must be specified unless restoring a model.")

        # Initialize attributes
        self.n_parameters = n_parameters
        self.n_wavelengths = pca_transform_matrix.shape[-1]
        self.n_pcas = pca_transform_matrix.shape[0]
        self.n_hidden = n_hidden
        self.architecture = [self.n_parameters] + self.n_hidden + [self.n_pcas]
        self.wavelengths = wavelengths

        # Transformations and constants with explicit dtype
        self.parameters_shift = jnp.array(parameters_shift if parameters_shift is not None else jnp.zeros(self.n_parameters), dtype=jnp.float32)
        self.parameters_scale = jnp.array(parameters_scale if parameters_scale is not None else jnp.ones(self.n_parameters), dtype=jnp.float32)
        self.pca_shift = jnp.array(pca_shift if pca_shift is not None else jnp.zeros(self.n_pcas), dtype=jnp.float32)
        self.pca_scale = jnp.array(pca_scale if pca_scale is not None else jnp.ones(self.n_pcas), dtype=jnp.float32)
        self.log_spectrum_shift = jnp.array(log_spectrum_shift if log_spectrum_shift is not None else jnp.zeros(self.n_wavelengths), dtype=jnp.float32)
        self.log_spectrum_scale = jnp.array(log_spectrum_scale if log_spectrum_scale is not None else jnp.ones(self.n_wavelengths), dtype=jnp.float32)
        self.pca_transform_matrix = jnp.array(pca_transform_matrix, dtype=jnp.float32)

        # Initialize or restore parameters
        key = jax.random.PRNGKey(0)  # Fixed seed for reproducibility
        if restore:
            self.params = self.initialize_params_from_restored_data()
        else:
            self.params = self.initialize_parameters(key)

        # Optimizer
        self.optimizer = optimizer or optax.adam(learning_rate=0.001)
        self.opt_state = self.optimizer.init(self.params)

    def initialize_parameters(self, key) -> Dict[str, Any]:
        """Initialize weights, biases, and activation parameters."""
        params = {}
        for i in range(len(self.architecture) - 1):
            n_in, n_out = self.architecture[i], self.architecture[i + 1]
            key, subkey = jax.random.split(key)
            params[f"W_{i}"] = jax.random.normal(subkey, (n_in, n_out), dtype=jnp.float32) * jnp.sqrt(2.0 / n_in)
            params[f"b_{i}"] = jnp.zeros((n_out,), dtype=jnp.float32)
            if i < len(self.architecture) - 2:  # No activation params for the last layer
                params[f"alpha_{i}"] = jax.random.normal(subkey, (n_out,), dtype=jnp.float32)
                params[f"beta_{i}"] = jax.random.normal(subkey, (n_out,), dtype=jnp.float32)
        return params

    @staticmethod
    def activation(self, x, alpha, beta):
        """Custom activation function with stability improvements."""
        epsilon = 1e-6
        return (beta + (1.0 - beta) * jax.nn.sigmoid(alpha * x + epsilon)) * x

    def forward(self, params, x):
        """Forward pass through the network."""
        for i in range(len(self.architecture) - 1):
            x = jnp.dot(x, params[f"W_{i}"]) + params[f"b_{i}"]
            if i < len(self.architecture) - 2:  # Apply activation to hidden layers
                x = self.activation(x, params[f"alpha_{i}"], params[f"beta_{i}"])
        return x

    def call(self, parameters):
        """Compute PCA coefficients."""
        normalized_params = (parameters - self.parameters_shift) / self.parameters_scale
        pca_coefficients = self.forward(self.params, normalized_params)
        return pca_coefficients * self.pca_scale + self.pca_shift

    def log_spectrum(self, parameters):
        """Transform PCA coefficients to log spectrum."""
        pca_coefficients = self.call(parameters)
        normalized_spectrum = jnp.dot(pca_coefficients, self.pca_transform_matrix)
        return normalized_spectrum * self.log_spectrum_scale + self.log_spectrum_shift

    def log_spectrum_(self, parameters):
        # Normalizing input parameters
        layers = [(parameters - self.parameters_shift) / self.parameters_scale]
        
        # Forward pass through the layers
        act = []
        for i in range(self.n_layers - 1):
            # Linear transformation
            act.append(jnp.dot(layers[-1], self.params[f"W_{i}"]) + self.params[f"b_{i}"])
            
            # Activation function
            alphas = self.params.get(f"alpha_{i}", None)
            betas = self.params.get(f"beta_{i}", None)

            if alphas is not None and betas is not None:
                # Reshaping for broadcasting if necessary
                alphas = alphas.reshape(1, -1) if len(alphas.shape) == 1 else alphas
                betas = betas.reshape(1, -1) if len(betas.shape) == 1 else betas

                layers.append((betas + (1.0 - betas) * jax.nn.sigmoid(alphas * act[-1])) * act[-1])
            else:
                layers.append(act[-1])  # Linear activation for the last layer

        # Final linear layer -> (normalized) PCA coefficients
        layers.append(jnp.dot(layers[-1], self.params[f"W_{self.n_layers - 1}"]) + self.params[f"b_{self.n_layers - 1}"])

        # Rescale the PCA coefficients
        pca_scale = self.pca_scale.reshape(1, -1) if len(self.pca_scale.shape) == 1 else self.pca_scale
        pca_shift = self.pca_shift.reshape(1, -1) if len(self.pca_shift.shape) == 1 else self.pca_shift

        return layers[-1] * pca_scale + pca_shift

    def update_emulator_parameters(self):
        """Convert JAX arrays to numpy for saving or external use."""
        return {key: jnp.array(value) for key, value in self.params.items()}

    def compute_loss_and_gradients_spectra(self, spectra, parameters, noise_floor):
        """Compute loss and gradients for spectra."""
        def loss_fn(params):
            predicted_spectra = jnp.exp(self.log_spectrum(parameters))
            return jnp.sqrt(jnp.mean(((predicted_spectra - spectra) / noise_floor) ** 2))
        
        loss, grads = jax.value_and_grad(loss_fn)(self.params)
        return loss, grads

    def training_step_spectra(self, spectra, parameters, noise_floor):
        """Perform a single training step for spectra."""
        loss, grads = self.compute_loss_and_gradients_spectra(spectra, parameters, noise_floor)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss

    def training_step_with_accumulated_gradients_spectra(self, spectra, parameters, noise_floor, accumulation_steps=10):
        """Train with accumulated gradients over multiple steps."""
        dataset = jax.tree_util.tree_map(lambda x: x.reshape(accumulation_steps, -1), (spectra, parameters, noise_floor))
        accumulated_loss = 0
        accumulated_grads = jax.tree_map(jnp.zeros_like, self.params)

        for batch in zip(*dataset):
            loss, grads = self.compute_loss_and_gradients_spectra(*batch)
            accumulated_grads = jax.tree_util.tree_map(jnp.add, accumulated_grads, grads)
            accumulated_loss += loss / accumulation_steps

        updates, self.opt_state = self.optimizer.update(accumulated_grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return accumulated_loss

    def compute_loss_and_gradients_pca(self, pca, parameters):
        """Compute loss and gradients for PCA coefficients."""
        def loss_fn(params):
            predicted_pca = self.call(parameters)
            return jnp.sqrt(jnp.mean((predicted_pca - pca) ** 2))
        
        loss, grads = jax.value_and_grad(loss_fn)(self.params)
        return loss, grads

    def training_step_pca(self, pca, parameters):
        """Perform a single training step for PCA."""
        loss, grads = self.compute_loss_and_gradients_pca(pca, parameters)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss

    def save(self, filename):
        """Save model parameters."""
        # Convert JAX arrays back to numpy for saving
        self.W_ = [jnp.array(self.params[f"W_{i}"]) for i in range(len(self.architecture) - 1)]
        self.b_ = [jnp.array(self.params[f"b_{i}"]) for i in range(len(self.architecture) - 1)]
        self.alphas_ = [jnp.array(self.params.get(f"alpha_{i}", None)) for i in range(len(self.architecture) - 2)]
        self.betas_ = [jnp.array(self.params.get(f"beta_{i}", None)) for i in range(len(self.architecture) - 2)]

        # Save all attributes as a list
        data = [
            self.W_, self.b_, self.alphas_, self.betas_, self.parameters_shift_,
            self.parameters_scale_, self.pca_shift_, self.pca_scale_,
            self.log_spectrum_shift_, self.log_spectrum_scale_, self.pca_transform_matrix_,
            self.n_parameters, self.n_wavelengths, self.wavelengths, self.n_pcas,
            self.n_hidden, self.n_layers, self.architecture
        ]
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def initialize_params_from_restored_data(self):
        """Initialize parameters using the restored data."""
        params = {}
        for i in range(len(self.W_)):
            params[f"W_{i}"] = jnp.array(self.W_[i])
            params[f"b_{i}"] = jnp.array(self.b_[i])
        for i in range(len(self.alphas_)):
            params[f"alpha_{i}"] = jnp.array(self.alphas_[i])
            params[f"beta_{i}"] = jnp.array(self.betas_[i])
        return params

    def restore(self, filename):
        """Restore model parameters."""
        with open(filename, 'rb') as f:
            (self.W_, self.b_, self.alphas_, self.betas_, self.parameters_shift_,
             self.parameters_scale_, self.pca_shift_, self.pca_scale_, self.log_spectrum_shift_,
             self.log_spectrum_scale_, self.pca_transform_matrix_, self.n_parameters,
             self.n_wavelengths, self.wavelengths, self.n_pcas, self.n_hidden,
             self.n_layers, self.architecture) = pickle.load(f)

        # Set attributes for restored model
        self.n_parameters = self.n_parameters
        self.n_wavelengths = self.n_wavelengths
        self.n_pcas = self.n_pcas
        self.n_hidden = self.n_hidden
        self.architecture = self.architecture


    def compute_loss_spectra(self, spectra, parameters, noise_floor):
        """Compute loss between predicted and true spectra."""
        predicted_spectra = jnp.exp(self.log_spectrum(parameters))
        return jnp.sqrt(jnp.mean(((predicted_spectra - spectra) / noise_floor) ** 2))

    def compute_loss_pca(self, pca, parameters):
        """Compute loss between predicted and true PCA coefficients."""
        predicted_pca = self.call(parameters)
        return jnp.sqrt(jnp.mean((predicted_pca - pca) ** 2))

    def compute_loss_and_gradients_log_spectra(self, log_spectra, parameters):
        """Compute loss and gradients for log spectra."""
        def loss_fn(params):
            predicted_log_spectra = self.log_spectrum(parameters)
            return jnp.sqrt(jnp.mean((predicted_log_spectra - log_spectra) ** 2))

        loss, grads = jax.value_and_grad(loss_fn)(self.params)
        return loss, grads

    def training_step_with_accumulated_gradients_pca(self, pca, parameters, accumulation_steps=10):
        """Train with accumulated gradients for PCA coefficients."""
        dataset = jax.tree_util.tree_map(lambda x: x.reshape(accumulation_steps, -1), (pca, parameters))
        accumulated_loss = 0
        accumulated_grads = jax.tree_map(jnp.zeros_like, self.params)

        for batch in zip(*dataset):
            loss, grads = self.compute_loss_and_gradients_pca(*batch)
            accumulated_grads = jax.tree_util.tree_map(jnp.add, accumulated_grads, grads)
            accumulated_loss += loss / accumulation_steps

        updates, self.opt_state = self.optimizer.update(accumulated_grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return accumulated_loss

    def training_step_log_spectra(self, log_spectra, parameters):
        """Perform a single training step for log spectra."""
        loss, grads = self.compute_loss_and_gradients_log_spectra(log_spectra, parameters)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return loss

    def training_step_with_accumulated_gradients_log_spectra(self, log_spectra, parameters, accumulation_steps=10):
        """Train with accumulated gradients for log spectra."""
        dataset = jax.tree_util.tree_map(lambda x: x.reshape(accumulation_steps, -1), (log_spectra, parameters))
        accumulated_loss = 0
        accumulated_grads = jax.tree_map(jnp.zeros_like, self.params)

        for batch in zip(*dataset):
            loss, grads = self.compute_loss_and_gradients_log_spectra(*batch)
            accumulated_grads = jax.tree_util.tree_map(jnp.add, accumulated_grads, grads)
            accumulated_loss += loss / accumulation_steps

        updates, self.opt_state = self.optimizer.update(accumulated_grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        return accumulated_loss
