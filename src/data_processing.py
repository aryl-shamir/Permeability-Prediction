import pandas as pd
import numpy as np


def load_well_data(path):
    """
    Load a well dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded well dataset.
    """

    return pd.read_csv(
        path,
        delimiter=";",
        na_values=-999.25
    )


def combine_wells(well_dfs, well_names):
    """
    Combine multiple well datasets and add well identifiers.

    Parameters
    ----------
    well_dfs : list
        List of well DataFrames.

    well_names : list
        List of well names corresponding to the DataFrames.

    Returns
    -------
    pandas.DataFrame
        Combined dataset with a 'Well Name' column.
    """

    all_wells = []

    for df, name in zip(well_dfs, well_names):
        df = df.copy()
        df["Well Name"] = name
        all_wells.append(df)

    combined_df = pd.concat(
        all_wells,
        axis=0,
        ignore_index=True
    )

    return combined_df


def create_logK(df):
    """
    Remove DEPTH and create the log10-transformed
    permeability target.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing DEPTH and K.

    Returns
    -------
    pandas.DataFrame
        Dataset without DEPTH and with logK.
    """

    df = df.copy()

    df = df.drop(
        columns=["DEPTH"]
    )

    df["logK"] = np.log10(
        df["K"]
    )

    return df