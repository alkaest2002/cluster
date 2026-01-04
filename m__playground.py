import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from lib.medoids.analyzer import KMedoidsAnalyzer
    return KMedoidsAnalyzer, Path, np, pd


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
def _(KMedoidsAnalyzer, cat_features):
    analyzer = KMedoidsAnalyzer(cat_features=cat_features)
    return (analyzer,)


@app.cell
def _(analyzer, df):
    best_model, best_k, dist_matrix, results_df = analyzer.run_optimization(df=df,n_clusters_min=2,n_clusters_max=30)
    return (dist_matrix,)


@app.cell
def _(dist_matrix):
    dist_matrix
    return


@app.cell
def _(Path, analyzer):
    with Path("./plot.svg").open("w") as f_out:
        f_out.write(analyzer.plot_metrics())
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
