import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from lib.paths import DATAPATH
    return DATAPATH, np, pd


@app.cell
def _(DATAPATH, pd):
    df = pd.read_excel(DATAPATH / "nomiac.xlsx", sheet_name=2, header=None)
    df.columns = [f"col_{c}" for c in df.columns]
    df["col_2"] = df["col_2"].where(~df["col_2"].isna(), df["col_1"])
    df["col_0"] = df["col_0"].ffill().fillna("dati generali")
    df = df.apply(lambda x: x.str.lower()).reset_index(drop=False, names="order_id")
    df_part1 = df.iloc[:, :4]
    df_part2 = df.iloc[:, 4:].fillna("")
    df
    return df_part1, df_part2


@app.cell
def _(df_part1, df_part2, pd):
    # BINARY VARS
    binary_idx = df_part2.col_3.str.contains("binaria")
    split_binary = (
        df_part2[binary_idx].col_3.str.split(" ", n=1, expand=True)
            .apply(lambda x: x.str.replace("=", ":").str.replace("(", "{").str.replace(")","}").str.replace("/", ","))
            .rename(columns={0:"col_3", 1: "col_4"})
    )

    df_binary = pd.concat([
        df_part1[binary_idx], 
        split_binary,
    ], axis=1)
    df_binary
    return binary_idx, df_binary


@app.cell
def _(df_part1, df_part2, pd):
    # CATEGORY
    category_idx = df_part2.col_3.str.contains("categoriale")
    cateory_values = (
        df_part2.loc[category_idx, pd.IndexSlice["col_4":"col_10"]]
            .apply(lambda x: x.str.replace("= ", ":").str.replace("=", ":"))
            .apply(lambda x: "{" + x.col_4 + "," + x.col_5 +"," +  x.col_6 +"," +  x.col_7 +"," +  x.col_8 +"," +  x.col_9 +"," +  x.col_10 + "}", axis=1)
    )
    cateory_values.name = "col_4"
    df_category = pd.concat([
        df_part1[category_idx], 
        df_part2.loc[category_idx, "col_3"],
        cateory_values,
    ], axis=1)
    df_category
    return category_idx, df_category


@app.cell
def _(binary_idx, category_idx, df_part1, df_part2, np, pd):
    # OTHER
    other_idx = ~np.logical_or(binary_idx, category_idx)
    df_other = pd.concat(
        [
            df_part1[other_idx],
            df_part2[other_idx]
        ]
    , axis=1)
    df_other
    return (df_other,)


@app.cell
def _(df_binary, df_category, df_other, pd):
    df_combo = pd.concat([df_binary, df_category, df_other], axis=0).sort_values(by="order_id").drop("order_id", axis=1)
    #df_combo.dropna(axis=1).to_json(DATAPATH / "out" / "codebook.json", index=False)
    return


@app.cell
def _(DATAPATH, pd):
    legend = pd.read_json(DATAPATH / "out" / "codebook.json")
    legend.to_csv(DATAPATH / "out" / "codebook.csv", index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
