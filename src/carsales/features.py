"""Functions used to clean and preprocess dataset.
"""
from typing import Optional

import pandas as pd


def remove_chars(data: pd.Series, chars: list[str]) -> pd.Series:
    """Removes characters specified in `chars` in each entry of `data`.

    Args:
        data (pd.Series):
            The data where specific characters should be
            deleted.
        chars (list[str]):
            The characters that should be removed in each entry of
            `data`.

    Returns:
        pd.Series:
            Cleaned data with removed characters.
    """
    for char in chars:
        data = data.str.replace(char, "")

    return data


def remove_prices(prices: pd.Series,
    price_min: float,
    price_max: float) -> pd.Series:
    """Removes unrealistic prices. This is specified by range of values.

    Args:
        prices (pd.Series):
            The original autos prices.
        price_min (float):
            Lower price bound.
        price_max (float):
            Upper price bound.

    Returns:
        pd.Series:
            The cleaned autos prices.
    """
    mask = prices.between(price_min, price_max)
    prices = prices[mask]

    return prices


def create_date_distrib(dates: pd.Series) -> pd.Series:
    """Creates a normalized statistical distribution of `dates`.

    Args:
        dates (pd.Series):
            The dates to be used for the statistical distribution.

    Returns:
        pd.Series:
            The normalized statistical distribution from `dates`.
    """
    dates = pd.to_datetime(dates)
    distrib = dates.dt.date.value_counts(normalize=True).sort_index()

    return distrib


def remove_registrations(
    autos: pd.DataFrame,
    min_acceptable_year: int,
    max_acceptable_year: int,
    inplace: Optional[bool] = False,
) -> Optional[pd.DataFrame]:
    """Removes listings which are associated to an unrealistic
    registration year. See the project notebook for more detail.

    Args:
        autos (pd.DataFrame):
            The listings dataset.
        min_acceptable_year (int):
            The lower bound year.
        max_acceptable_year (int):
            The upper bound year.
        inplace (Optional[bool], optional):
            Whether or not to operate in-place. Defaults to False.

    Returns:
        Optional[pd.DataFrame]:
            If `inplace` is set to `True`, the function directly
            operates on the existing `autos` dataset and does not return
            any dataframe. If `inplace` is set to `False`, a new
            dataframe is returned.
    """
    mask = ~autos["registration_year"].between(min_acceptable_year,
                                               max_acceptable_year)
    if inplace:
        autos.drop(autos[mask].index, inplace=inplace)
    else:
        return autos.drop(autos[mask].index, inplace=inplace)


def compute_brands_avg_feature(
    autos: pd.DataFrame, brands: pd.Index, feature: str) -> dict[str, float]:
    """Computes an average `feature` for each brand in `brands`.

    Args:
        autos (pd.DataFrame):
            The listings dataset.
        brands (pd.Index):
            The car brands we want to compute the average price.
        feature (str):
            The feature name whose mean we want to compute.

    Returns:
        dict[str, float]:
            Average prices for each brand.
    """
    avg_feature = {}

    for brand in brands:
        avg_feature[brand] = round(
            autos.loc[autos["brand"] == brand, feature].mean(), 2
        )

    return avg_feature
