"""Functions used to clean and preprocess dataset.
"""
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


def remove_prices(
    prices: pd.Series, price_min: float, price_max: float) -> pd.Series:
    """Removes unrealistic prices. This is specified by range of values.

    Args:
        prices (pd.Series):
            The original autos prices.
        price_min (float):
            Lower price bound.
        price_max (float):
            Higher price bound.

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
