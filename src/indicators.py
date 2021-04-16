import pandas as pd
from matplotlib import pylab as plt


def add_sma(df, ticker='JPM', n_days=30):
    df.loc[:, f'{ticker}_sma_{n_days}'] = df[ticker].rolling(n_days, closed='right', min_periods=n_days).mean()
    return df


def add_sma_ratio(df, ticker='JPM', n_days=30):
    sma_col = f'{ticker}_sma_{n_days}'
    if not sma_col in df:
        add_sma(df, ticker=ticker, n_days=n_days)
    df.loc[:, f'{ticker}_sma_ratio_{n_days}'] = df[ticker]/df[sma_col] - 1
    return df


def add_std(df, ticker='JPM', n_days=10):
    df.loc[:, f'{ticker}_std_{n_days}'] = df[ticker].rolling(n_days, closed='right', min_periods=n_days).std()
    return df


def add_bb(df, ticker='JPM', n_days=10):
    std_col = f'{ticker}_std_{n_days}'
    sma_col = f'{ticker}_sma_{n_days}'
    if not std_col in df:
        add_std(df, ticker=ticker, n_days=n_days)
    if not sma_col in df:
        add_sma(df, ticker=ticker, n_days=n_days)
    bw = 2 * df[std_col]
    df.loc[:, f'{ticker}_bb_upper_{n_days}'] = df[sma_col] + bw
    df.loc[:, f'{ticker}_bb_lower_{n_days}'] = df[sma_col] - bw
    return df


def add_bb_ratio(df, ticker='JPM', n_days=10):
    std_col = f'{ticker}_std_{n_days}'
    sma_col = f'{ticker}_sma_{n_days}'
    if not std_col in df:
        add_std(df, ticker=ticker, n_days=n_days)
    if not sma_col in df:
        add_sma(df, ticker=ticker, n_days=n_days)
    df[f'{ticker}_bb_ratio_{n_days}'] = (df[ticker] - df[sma_col]) / (2 * df[std_col])
    return df


def add_momentum(df, ticker='JPM', n_days=10):
    df.loc[:, f'{ticker}_momentum_{n_days}'] = df[ticker].pct_change(periods=n_days)
    return df


def add_ema(df, ticker='JPM', n_days=10):
    df.loc[:, f'{ticker}_ema_{n_days}'] = df[ticker].ewm(span=n_days).mean()
    return df


def add_macd(df, ticker='JPM', n_days1=12, n_days2=26):
    ema1_col = f'{ticker}_ema_{n_days1}'
    ema2_col = f'{ticker}_ema_{n_days2}'
    if not ema1_col in df:
        add_ema(df, ticker=ticker, n_days=n_days1)
    if not ema2_col in df:
        add_ema(df, ticker=ticker, n_days=n_days2)
    df.loc[:, f'{ticker}_macd_{n_days1}_{n_days2}'] = df[ema1_col] - df[ema2_col]
    return df


def add_macd_signal_line(df, ticker='JPM', n_days1=12, n_days2=26, macd_ema_span=9):
    macd_col = f'{ticker}_macd_{n_days1}_{n_days2}'
    if not macd_col in df:
        add_macd(df, ticker=ticker, n_days1=n_days1, n_days2=n_days2)
    df.loc[:, f'{ticker}_macd_signal_{n_days1}_{n_days2}_{macd_ema_span}'] = df[macd_col].ewm(span=macd_ema_span).mean()
    return df


def add_stochastic_oscillator(df, ticker='JPM', n_days=14):
    high = df[ticker].rolling(n_days, closed='right', min_periods=n_days).max()
    low = df[ticker].rolling(n_days, closed='right', min_periods=n_days).min()
    df.loc[:, f'{ticker}_so_{n_days}'] = (df[ticker] - low) / (high - low)
    return df
