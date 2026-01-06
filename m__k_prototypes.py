import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd

    from lib.paths import DATAPATH
    from lib.pipes import preprocess_k_prototypes
    return DATAPATH, pd, preprocess_k_prototypes


@app.cell
def _(DATAPATH, pd):
    df = pd.read_parquet(DATAPATH / "out" / "from_synth_clean.parquet")
    df
    return (df,)


@app.cell
def _(df, preprocess_k_prototypes):
    pipe = preprocess_k_prototypes.get_pipe(df)
    results = pipe.fit_transform(df)
    results = results.rename(lambda x: x.split("__", maxsplit=1)[-1], axis='columns')
    results
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
