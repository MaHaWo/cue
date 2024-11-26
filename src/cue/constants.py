import jax.numpy as jnp

# Constants file for the JAX-Flax project
# This file provides a centralized location for storing commonly used constants throughout the project.

# Physical constants
LIGHT_SPEED = 2.998e8  # Speed of light in meters per second
PLANCK_CONSTANT = 6.626e-34  # Planck's constant in J s
BOLTZMANN_CONSTANT = 1.381e-23  # Boltzmann constant in J/K

# Model default parameters
DEFAULT_CONTINUUM_PARAMS = jnp.array([21.5, 14.85, 6.45, 3.15, 4.55, 0.7, 0.85, 49.58, 10**2.5, -0.85, -0.134, -0.134])
DEFAULT_LINE_PARAMS = jnp.array([25.1, 16.9, 5.5, 3.3, 5.0, 0.6, 0.9, 51.0, 10**2.3, -0.9, -0.14, -0.14])

# PCA related constants
PCA_NUM_COMPONENTS_CONT = 10  # Number of PCA components for continuum model
PCA_NUM_COMPONENTS_LINE = 8   # Number of PCA components for line model

# Normalization values for PCA models (example values)
PCA_CONT_MEAN = jnp.array([0.5] * PCA_NUM_COMPONENTS_CONT)
PCA_CONT_STD = jnp.array([0.1] * PCA_NUM_COMPONENTS_CONT)
PCA_LINE_MEAN = jnp.array([0.4] * PCA_NUM_COMPONENTS_LINE)
PCA_LINE_STD = jnp.array([0.2] * PCA_NUM_COMPONENTS_LINE)

