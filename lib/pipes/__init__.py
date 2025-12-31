from sklearn import set_config

# Set the global configuration for scikit-learn to output pandas DataFrames
set_config(transform_output="pandas")
