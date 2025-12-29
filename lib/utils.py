import numpy as np
import pandas as pd
from numpy.typing import NDArray


def greet(name: str) -> str:
    """Just a simple greeting function.

    Args:
        name (str): The name of the person to greet.

    Returns:
        str: A greeting message.

    """
    return f"Hello, {name}!"


def get_columns_indices_from_regex(df: pd.DataFrame, regex: str) ->NDArray[np.int_]:
    """Get the indices of columns in a DataFrame that match a given regex pattern.
    
    Args:
        df (pd.DataFrame): The DataFrame to search.
        regex (str): The regex pattern to match column names.
    
    Returns:
        NDArray[np.int_]: An array of indices of matching columns.
    """
    # Create a boolean mask where column names match the regex
    condition: pd.Series = df.columns.str.contains(regex, regex=True)
    
    # Return the indices of the columns that match the regex
    return np.asarray(condition).nonzero()[0]
