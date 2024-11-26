import jax.numpy as jnp

# Utility functions for the JAX-Flax project

# Define a function to compute mean squared error (MSE)
def mean_squared_error(y_true, y_pred):
    """
    Computes the mean squared error between the true and predicted values.
    Args:
        y_true (jnp.ndarray): True values.
        y_pred (jnp.ndarray): Predicted values.
    Returns:
        jnp.ndarray: The mean squared error.
    """
    return jnp.mean((y_true - y_pred) ** 2)

# Define a function for computing the root mean squared error (RMSE)
def root_mean_squared_error(y_true, y_pred):
    """
    Computes the root mean squared error between the true and predicted values.
    Args:
        y_true (jnp.ndarray): True values.
        y_pred (jnp.ndarray): Predicted values.
    Returns:
        jnp.ndarray: The root mean squared error.
    """
    return jnp.sqrt(mean_squared_error(y_true, y_pred))

# Define a function to compute the R-squared (coefficient of determination)
def r_squared(y_true, y_pred):
    """
    Computes the R-squared value to determine the quality of the fit.
    Args:
        y_true (jnp.ndarray): True values.
        y_pred (jnp.ndarray): Predicted values.
    Returns:
        jnp.ndarray: The R-squared value.
    """
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
