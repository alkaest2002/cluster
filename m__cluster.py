import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd

    from lib.paths import DATAPATH
    return DATAPATH, pd


@app.cell
def _(DATAPATH, pd):
    df = pd.read_feather(DATAPATH / "out" / "from_pipe.feather")
    df["cat__cat_dept"].cat.categories
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
