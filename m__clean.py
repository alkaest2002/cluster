import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import pandas as pd

    from lib.paths import DATAPATH
    return DATAPATH, Path, pd


@app.cell
def _(DATAPATH, Path, pd):
    name_of_file = Path("synth") 
    df = pd.read_csv(DATAPATH / name_of_file.with_suffix(".csv"))
    return df, name_of_file


@app.cell
def _(df, pd):
    ##################################################################################################################
    # Columns names and string names should be lower case
    ##################################################################################################################
    df.columns = df.columns.str.lower()
    df_clean = df.apply(lambda s: s.str.lower() if pd.api.types.is_string_dtype(s) else s)


    df_clean
    return (df_clean,)


@app.cell
def _(DATAPATH, df_clean, name_of_file):
    df_clean.to_csv(DATAPATH / f"{name_of_file}_clean.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
