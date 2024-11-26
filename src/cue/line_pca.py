import jax
import jax.numpy as jnp
import pickle

# Load line PCA parameters from file
with open("data/pca_line_new_Ne4.pkl", "rb") as f:
    pca_line_params = pickle.load(f)

# Define normalization utility function for line PCA parameters
def normalize_line_params(line_params, mean, std):
    """
    Normalizes line parameters using the provided mean and standard deviation.
    Args:
        line_params (jnp.ndarray): Input line parameters to be normalized.
        mean (float or jnp.ndarray): Mean value(s) for normalization.
        std (float or jnp.ndarray): Standard deviation value(s) for normalization.
    Returns:
        jnp.ndarray: Normalized line parameters.
    """
    return (line_params - mean) / std

# Define rescaling utility function for line PCA parameters
def rescale_line_params(normalized_params, mean, std):
    """
    Rescales normalized line parameters back to original scale.
    Args:
        normalized_params (jnp.ndarray): Normalized line parameters.
        mean (float or jnp.ndarray): Mean value(s) used for normalization.
        std (float or jnp.ndarray): Standard deviation value(s) used for normalization.
    Returns:
        jnp.ndarray: Rescaled line parameters.
    """
    return (normalized_params * std) + mean

# Load mean and std from pca parameters if available
#mean_line = pca_line_params.get('mean')
#std_line = pca_line_params.get('std')

# Dummy data for testing line PCA normalization
#x_line_dummy = jnp.array([[25.0, 16.5, 5.0, 3.2, 4.9, 0.75, 0.88, 50.0, 10**2.4, -0.87, -0.13, -0.13]])

# Normalize line parameters
#normalized_line_params = normalize_line_params(x_line_dummy, mean_line, std_line)

# Rescale back to original line parameters
#rescaled_line_params = rescale_line_params(normalized_line_params, mean_line, std_line)
