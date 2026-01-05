import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import importlib
    import lib.k_medoids
    importlib.reload(lib.k_medoids)

    from pathlib import Path
    from lib.k_medoids.optimizer import Optimizer
    import numpy as np
    import pandas as pd
    from lib.utils import PathUtils
    return Optimizer, Path, PathUtils, np, pd


@app.cell
def _(np, pd):
    def create_sample_data(n_samples: int = 300) -> pd.DataFrame:
        """Create sample mixed-type data for demonstration purposes.

        Generates a DataFrame with numerical and categorical features including
        age, income, score, gender, membership level, and activity status.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            DataFrame containing the generated sample data.
        """
        # Use random seed generator for reproducibility
        np.random.seed(42)
        data = {
            "age": np.random.randint(20, 70, n_samples),
            "income": np.random.normal(60000, 15000, n_samples),
            "score": np.random.randint(1, 100, n_samples),
            "gender": np.random.choice(["Male", "Female", "Other"], n_samples),
            "membership": np.random.choice(["Basic", "Silver", "Gold"], n_samples),
            "active": np.random.choice([True, False], n_samples)
        }
        return pd.DataFrame(data)
    return (create_sample_data,)


@app.cell
def _(create_sample_data):
    df = create_sample_data()
    cat_features = df.select_dtypes(include="object").columns.to_list()
    return cat_features, df


@app.cell
def _(Optimizer, cat_features):
    optimizer = Optimizer(cat_features=cat_features)
    return (optimizer,)


@app.cell
def _(df, optimizer):
    best_model,dist_matrix, results_df = optimizer.optimize(df=df,n_clusters_min=2,n_clusters_max=30)
    return (best_model,)


@app.cell
def _(best_model):
    best_model.n_clusters
    return


@app.cell
def _(Path, PathUtils, optimizer):
    with Path(PathUtils.get_output_path() /"./plot.svg").open("w") as f_out:
        f_out.write(optimizer.get_plots()[0])
    return


@app.cell
def _(optimizer):
    optimizer.get_analysis()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
