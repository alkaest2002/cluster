import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd

    from sklearn import set_config
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    from lib.paths import DATAPATH
    from lib.utils import get_columns_indices_from_regex


    set_config(transform_output="pandas")
    return (
        ColumnTransformer,
        DATAPATH,
        Pipeline,
        StandardScaler,
        make_pipeline,
        pd,
    )


@app.cell
def _(DATAPATH, pd):
    df = pd.read_parquet(DATAPATH / "out" / "from_synth_clean.parquet", engine='fastparquet')
    df.head()
    return (df,)


@app.cell
def _(ColumnTransformer, Pipeline, StandardScaler, df, make_pipeline):
    ####################################################################################################################################
    # DEFINE TYPE OF METRICS
    ####################################################################################################################################
    num_cols = [df.columns.get_loc(c) for c in df.select_dtypes(exclude="object")]
    cat_cols = [df.columns.get_loc(c) for c in df.select_dtypes(include="object")]


    ####################################################################################################################################
    # DEFINE PIPELINES FOR DIFFERENT METRICS
    ####################################################################################################################################
    num_steps = [
        ("scaler", StandardScaler()),
    ]
    num = Pipeline(steps=num_steps)
    num_pipe =  ("num", num, num_cols)


    ####################################################################################################################################
    # DEFINE GENERAL PIPELINE
    ####################################################################################################################################
    preprocessor = ColumnTransformer([num_pipe], remainder="passthrough")
    pipe = make_pipeline(preprocessor)


    ####################################################################################################################################
    # APPLY GENERAL PIPELINE
    ####################################################################################################################################
    transformed = pipe.fit_transform(df)

    transformed.head()
    return (transformed,)


@app.cell
def _(DATAPATH, transformed):
    transformed.to_parquet(DATAPATH / "out" / "from_pipe.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
