import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


app._unparsable_cell(
    r"""
    from pathlib import Path

    import pandas as pd
    import numpy as np

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import 

    from lib.paths import DATAPATH
    from lib.utils import get_columns_indices_from_regex

    """,
    name="_"
)


@app.cell
def _(DATAPATH, get_columns_indices_from_regex, pd):
    df = pd.read_csv(DATAPATH / "synth.csv")
    get_columns_indices_from_regex(df, r"num_")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
