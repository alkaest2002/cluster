import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd

    from sklearn import set_config
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

    from lib.paths import DATAPATH
    from lib.utils import get_columns_indices_from_regex


    set_config(transform_output="pandas")
    return (
        ColumnTransformer,
        DATAPATH,
        OneHotEncoder,
        OrdinalEncoder,
        Pipeline,
        StandardScaler,
        get_columns_indices_from_regex,
        make_pipeline,
        pd,
    )


@app.cell
def _(DATAPATH, pd):
    df = pd.read_csv(DATAPATH / "synth.csv")
    df.head()
    return (df,)


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    Pipeline,
    StandardScaler,
    df,
    get_columns_indices_from_regex,
    make_pipeline,
    pd,
):
    ####################################################################################################################################
    # DEFINE TYPE OF METRICS
    ####################################################################################################################################
    num_cols = get_columns_indices_from_regex(df, "num_")
    cat_cols = get_columns_indices_from_regex(df, "cat_")
    nom_cols = get_columns_indices_from_regex(df, "nom_")


    ####################################################################################################################################
    # DEFINE PIPELINES FOR DIFFERENT METRICS
    ####################################################################################################################################
    num_steps = [("scaler", StandardScaler())]
    num = Pipeline(steps=num_steps)
    num_pipe =  ("num", num, num_cols)

    cat_steps = [("ord_encoder", OrdinalEncoder())]
    cat = Pipeline(steps=cat_steps)
    cat_pipe = ("cat", cat, cat_cols)

    nom_steps = [("ohe_encoder", OneHotEncoder(sparse_output=False))]
    nom = Pipeline(steps=nom_steps)
    nom_pipe = ("nom", nom, nom_cols)


    ####################################################################################################################################
    # DEFINE GENERAL PIPELINE
    ####################################################################################################################################
    preprocessor = ColumnTransformer([num_pipe, cat_pipe, nom_pipe])
    pipe = make_pipeline(preprocessor)


    ####################################################################################################################################
    # APPLY GENERAL PIPELINE
    ####################################################################################################################################
    transformed = pipe.fit_transform(df)

    # Access the categories/labels
    cat_encoder = preprocessor.named_transformers_['cat']['ord_encoder']

    # Manage categorical metrics
    for i, col in enumerate(transformed.filter(regex=r"cat_")):
       # Convert float to categorical
        transformed[col] = pd.Categorical(transformed[col])
        # Add category labels
        transformed[col] = transformed[col].cat.rename_categories(cat_encoder.categories_[i])

    transformed.head()
    return (transformed,)


@app.cell
def _(DATAPATH, transformed):
    transformed.to_feather(DATAPATH / "out" / "from_pipe.feather")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
