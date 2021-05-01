from pathlib import Path
import pandas as pd
import numpy as np
import logging
import pickle
import json
import progressbar

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SingleTickerPipeline:
    def __init__(
        self,
        target="price",
        target_type="sequence",
        model_seq_len=30,
        max_overlap=20,
        train_periods=[
            ("2000-01-01", "2006-12-31"),
            ("2009-01-01", "2018-12-31"),
        ],
        test_periods=[
            ("2007-01-01", "2008-12-31"),
            ("2019-01-01", "2021-04-01"),
        ],
        normalization_method="log",
        lookback_period=200,
        cross_validation_folds=5,
        data_path=Path(__file__).absolute().parent.parent.joinpath("data/feature_selected"),
        output_path=Path(__file__).absolute().parent.parent.joinpath("data/model_data"),
    ):
        """
        Parameters
        ----------
        target: str
            target can be "price", "return" or "log_return"
        target_type: str
            target type can be "single" for single point-in-time prediction 
            or "sequence" for sequence prediction (predicts a sequence of target shifted one day into the future)
            if single, output y shape is (N, 1)
            if sequence, output y shape is (N, model_seq_len, 1)
        model_seq_len: int
            model sequence length specifies the sequence length of each input sample. 
            E.g. 30 means using the past 30 days's historical data to predict the next day
        max_overlap: int
            maximum number of overlapping days between two sequences. Will be capped at model_seq_len - 1
            if it is larger than model_seq_len
        train_periods: list(tuple(str, str))
            training periods is a list of tuples, each tuple has a start date and an end date. 
            Data from all training periods are put together
            Note that training periods will be further divided into time series cross validation
        test_periods: list(tuple(str, str))
            similar to training periods
        normalization_method: str
            how features are normalized within each sequence
            None: no normalization performed
            "log": feature x is transformed into sign(x) * log(1 + |x|)
            "quantile": feature x is transformed into (x - P50) / (P75 - P25), where P25, P50 and P75 are 
                the 25th, 50th and 75th quantile of x in the past lookback_period records (if available)
        lookback_period: int
            number of records from the past used to estimate quantiles, only used if normalization_method is set to "quantile"
        cross_validation_folds: int
            number of folds for rolling cross validation
        data_path: str or pathlib.Path
            path to the input data directory. Default: project_root/data/feature_selected
        output_path: str or pathlib.Path
            root path to store the output data. Default: project_root/data/model_data
        """
        self.target = target
        self.target_col = "adj_close" if target == 'price' else target
        self.target_type = target_type
        self.model_seq_len = model_seq_len
        self.max_overlap = min(model_seq_len - 1, max_overlap)
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.normalization_method = normalization_method
        self.lookback_period = lookback_period
        self.cross_validation_folds = cross_validation_folds
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        
        # internal attributes
        self._df = None
        self._ticker = None
        self._feature_cols = None
        self._train_out = None
        self._test_out = None
        self._seq_dist = self.model_seq_len - self.max_overlap
        self._save_path = None
        
    def load_input(self, ticker=None):
        ticker = ticker or self._ticker
        data_file = self.data_path.joinpath(f"{ticker}.csv")
        LOG.info(f"Reading data from {data_file.as_posix()}...")
        df = pd.read_csv(data_file).drop('price', axis=1, errors="ignore").sort_values("date", ascending=True)
        if self.target == "price":
            df.loc[:, "target"] = df['adj_close']
        else: 
            if self.target == "return":
                return_col = get_return_col(df, log=False)
            if self.target == "log_return":
                return_col = get_return_col(df, log=True)
            df.loc[:, "target"] = df[return_col]
            df.drop(['adj_close'])
        self._feature_cols = df.drop(['date', 'target'], axis=1).columns.tolist()
        
        # match the target with data from the previous days
        df = pd.concat(
            [
                df[['target', 'date']].rename({'date': 'prediction_date'}, axis=1).iloc[1:, :].reset_index(drop=True),
                df[self._feature_cols + ["date"]].iloc[:-1, :].reset_index(drop=True)
            ],
            axis=1
        )
        self._df = df

    def get_xy_arr(self, dfs, seq_dist=None):
        seq_dist = seq_dist or self._seq_dist
        arrays = {"x": [], "y": [], "prediction_date": [], "N": 0}
        for df in dfs:
            N = df.shape[0]
            if N >= self.model_seq_len:
                for i in range((N - self.model_seq_len) // seq_dist):
                    feature_subdf = df[self._feature_cols].iloc[(N - (i * seq_dist + self.model_seq_len)):(N - i * seq_dist)]
                    target_col_copy = feature_subdf[[self.target_col]]
                    if self.normalization_method == "quantile":
                        feature_quantiles = df[self._feature_cols].iloc[
                            max(0, (N - (i * seq_dist + self.lookback_period))):(N - i * seq_dist)
                        ].quantile([0.25, 0.5, 0.75])
                        p25 = feature_quantiles.loc[0.25, :]
                        p50 = feature_quantiles.loc[0.50, :]
                        p75 = feature_quantiles.loc[0.75, :]
                        feature_subdf = ((feature_subdf - p50) / (p75 - p25))
                    elif self.normalization_method == "log":
                        feature_subdf = np.sign(feature_subdf) * np.log1p(np.abs(feature_subdf))
                    
                    feature_subdf = pd.concat([target_col_copy, feature_subdf], axis=1).replace([-np.inf, np.inf], np.nan).fillna(0)
                    arrays["x"].append(feature_subdf.values)
                    if self.target_type == "sequence":
                        arrays["y"].append(df[["target"]].iloc[(N - (i * seq_dist + self.model_seq_len)):(N - i * seq_dist)].values)
                    elif self.target_type == "single":
                        arrays["y"].append([df["target"].iloc[(N - i * seq_dist) - 1]])
                    else:
                        raise KeyError("Unknown target_type: target_type must be one of 'sequence' or 'single'!")
                    arrays["prediction_date"].append([df["prediction_date"].iloc[(N - i * seq_dist) - 1]])
                    arrays["N"] += 1
        arrays["x"] = np.array(arrays['x'][::-1])
        arrays["y"] = np.array(arrays['y'][::-1])
        arrays["prediction_date"] = np.array(arrays['prediction_date'][::-1])
        return arrays

    def create_train_array(self):
        LOG.info("Making training arrays...")
        train_dfs = get_period_data(self._df, self.train_periods)
        train_xy_arrs = self.get_xy_arr(train_dfs)
        LOG.info(f"  Training has {train_xy_arrs['N']} sequences of length {self.model_seq_len}.")
        
        LOG.info(f"Making {self.cross_validation_folds} validation folds...")
        train_val_distance = int(np.ceil(self.model_seq_len / self._seq_dist))
        fold_size = (train_xy_arrs["N"] - train_val_distance) // (self.cross_validation_folds + 1)
        
        LOG.info(f"  Generating folds with fold_size={fold_size} and distance between train and validation being {train_val_distance}")
        folds = {}
        for i in range(self.cross_validation_folds):
            train_end_ind = fold_size * (i + 1)
            val_begin_ind = fold_size * (i + 1) + train_val_distance
            val_end_ind = val_begin_ind + fold_size
            fold_arrs = {
                "train":{
                    "x": train_xy_arrs["x"][:train_end_ind],
                    "y": train_xy_arrs["y"][:train_end_ind],
                    "prediction_date": train_xy_arrs["prediction_date"][:train_end_ind],
                },
                "valid":{
                    "x": train_xy_arrs["x"][val_begin_ind:val_end_ind],
                    "y": train_xy_arrs["y"][val_begin_ind:val_end_ind],
                    "prediction_date": train_xy_arrs["prediction_date"][val_begin_ind:val_end_ind],
                },
            }
            folds[i] = fold_arrs
            LOG.info(f"    Fold {i} shapes:")
            for sample in fold_arrs:
                LOG.info(f"      x: {fold_arrs[sample]['x'].shape}, y: {fold_arrs[sample]['y'].shape}")
        folds["_all_"] = {'x': train_xy_arrs['x'], 'y': train_xy_arrs['y']}
        self._train_out = folds
        
    def create_test_array(self):
        LOG.info("Making testing arrays...")
        test_dfs = get_period_data(self._df, self.test_periods)
        test_xy_arrs = self.get_xy_arr(test_dfs, seq_dist=1)
        LOG.info(f"  Testing has {test_xy_arrs['N']} sequences of length {self.model_seq_len}.")
        self._test_out = test_xy_arrs
    
    def create_arrays(self):
        self.create_train_array()
        self.create_test_array()
    
    def write_data(self):
        self._save_path = self.output_path.joinpath(self._ticker)
        LOG.info(f"Saving generated data at {self._save_path.as_posix()}...")
        if not self._save_path.exists():
            LOG.info(f"  Directory doesn't exist, making directory...")
            self._save_path.mkdir(parents=True)
        LOG.info("  Writing train folds...")
        write_pickle_file(self._train_out, self._save_path.joinpath("train.pkl"))
        LOG.info("  Writing test arrays...")
        write_pickle_file(self._test_out, self._save_path.joinpath("test.pkl"))
        
    
    def prepare_data(self, ticker):
        self._ticker = ticker
        self.load_input()
        self.create_arrays()
        self.write_data()
        
    def load_data(self, ticker):
        self._ticker = ticker
        self._save_path = self.output_path.joinpath(self._ticker)
        LOG.info(f"Loading generated data from {self._save_path.as_posix()}...")
        if not self._save_path.exists():
            raise FileNotFoundError("Directory doesn't exist, can't load data!")
        LOG.info("  Loading train folds...")
        self._train_out = load_pickle_file(self._save_path.joinpath("train.pkl"))
        LOG.info("  Loading test arrays...")
        self._test_out = load_pickle_file(self._save_path.joinpath("test.pkl"))
    
    def print_train_shapes(self):
        print(
            json.dumps({
                i: {
                    s: (
                        {k: str(v.shape) for k, v in arr.items()} 
                        if isinstance(arr, dict) else (
                            str(arr.shape) if hasattr(arr, "shape") else arr
                        )
                    ) 
                    for s, arr in fold.items()
                } for i, fold in self._train_out.items()
            }, sort_keys=False, indent=4)
        )
        
    def print_test_shapes(self):
        print(
            json.dumps(
                {k: str(v.shape) if hasattr(v, "shape") else v for k, v in self._test_out.items()},
                sort_keys=True, indent=4
            )
        )
    
def get_return_col(df, log=False):
    price_rat = df['adj_close'] / df['adj_close'].shift(-1)
    if log:
        return_col_name = "log_return"
        return_col_value = np.log(price_rat)
    else:
        return_col_name = "return"
        return_col_value = price_rat - 1
    df.loc[:, return_col_name] = return_col_value
    return return_col_name


def get_period_data(df, periods, date_col="date"):
    dfs_by_period = [
        df[
            pd.to_datetime(df[date_col]).between(pd.to_datetime(period[0]), pd.to_datetime(period[1]))
        ].sort_values(date_col, ascending=True) for period in periods
    ]
    return dfs_by_period


def write_pickle_file(obj, file):
    with Path(file).open('wb') as pkl_file:
        pickle.dump(obj, pkl_file, protocol=4)


def load_pickle_file(file):
    with Path(file).open('rb') as pkl_file:
        obj = pickle.load(pkl_file)
    return obj


class MultiTickerPipeline(SingleTickerPipeline):
    def __init__(
        self,
        target="price",
        target_type="sequence",
        model_seq_len=30,
        max_overlap=20,
        train_periods=[
            ("2000-01-01", "2006-12-31"),
            ("2009-01-01", "2018-12-31"),
        ],
        test_periods=[
            ("2007-01-01", "2008-12-31"),
            ("2019-01-01", "2021-04-01"),
        ],
        normalization_method="log",
        lookback_period=200,
        cross_validation_folds=5,
        data_path=Path(__file__).absolute().parent.parent.joinpath("data/feature_selected"),
        output_path=Path(__file__).absolute().parent.parent.joinpath("data/model_data"),
    ):
        """
        Parameters
        ----------
        target: str
            target can be "price", "return" or "log_return"
        target_type: str
            target type can be "single" for single point-in-time prediction 
            or "sequence" for sequence prediction (predicts a sequence of target shifted one day into the future)
            if single, output y shape is (N, 1)
            if sequence, output y shape is (N, model_seq_len, 1)
        model_seq_len: int
            model sequence length specifies the sequence length of each input sample. 
            E.g. 30 means using the past 30 days's historical data to predict the next day
        max_overlap: int
            maximum number of overlapping days between two sequences. Will be capped at model_seq_len - 1
            if it is larger than model_seq_len
        train_periods: list(tuple(str, str))
            training periods is a list of tuples, each tuple has a start date and an end date. 
            Data from all training periods are put together
            Note that training periods will be further divided into time series cross validation
        test_periods: list(tuple(str, str))
            similar to training periods
        normalization_method: str
            how features are normalized within each sequence
            None: no normalization performed
            "log": feature x is transformed into sign(x) * log(1 + |x|)
            "quantile": feature x is transformed into (x - P50) / (P75 - P25), where P25, P50 and P75 are 
                the 25th, 50th and 75th quantile of x in the past lookback_period records (if available)
        lookback_period: int
            number of records from the past used to estimate quantiles, only used if normalization_method is set to "quantile"
        cross_validation_folds: int
            number of folds for rolling cross validation
        data_path: str or pathlib.Path
            path to the input data directory. Default: project_root/data/feature_selected
        output_path: str or pathlib.Path
            root path to store the output data. Default: project_root/data/model_data
        """
        self.target = target
        self.target_col = "adj_close" if target == 'price' else target
        self.target_type = target_type
        self.model_seq_len = model_seq_len
        self.max_overlap = min(model_seq_len - 1, max_overlap)
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.normalization_method = normalization_method
        self.lookback_period = lookback_period
        self.cross_validation_folds = cross_validation_folds
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        
        # internal attributes
        self._single_pipelines = None
        self._df = None
        self._tickers = None
        self._feature_cols = None
        self._train_out = None
        self._test_out = None
        self._seq_dist = self.model_seq_len - self.max_overlap
        self._save_path = None
        self._train_xy_arrays = None
        self._test_xy_arrays = None
    
    def load_input(self, tickers):
        self._single_pipelines = {}
        for ticker in tickers:
            single_pipeline = SingleTickerPipeline(
                self.target,
                self.target_type,
                self.model_seq_len,
                self.max_overlap,
                self.train_periods,
                self.test_periods,
                self.normalization_method,
                self.lookback_period,
                self.cross_validation_folds,
                self.data_path,
                self.output_path,
            )
            single_pipeline.load_input(ticker)
            self._single_pipelines[ticker] = single_pipeline
    
    def create_arrays(self):
        LOG.info("Making training arrays...")
        train_xy_arrays = {}
        test_xy_arrays = {}
        for ticker in progressbar.progressbar(self._single_pipelines):
            single_pipeline = self._single_pipelines[ticker]
            train_dfs = get_period_data(single_pipeline._df, self.train_periods)
            train_xy_arrays[ticker] = single_pipeline.get_xy_arr(train_dfs)
            train_xy_arrays[ticker]['ticker'] = np.repeat(ticker, train_xy_arrays[ticker]['y'].shape[0])
            
            test_dfs = get_period_data(single_pipeline._df, self.test_periods)
            test_xy_arrays[ticker] = single_pipeline.get_xy_arr(test_dfs, seq_dist=1)
            test_xy_arrays[ticker]['ticker'] = np.repeat(ticker, test_xy_arrays[ticker]['y'].shape[0])
            
        # combine results
        
        self._train_xy_arrays = {}
        self._test_xy_arrays = {}
        for key in "x", "y", "prediction_date", "ticker":
            self._train_xy_arrays[key] = np.concatenate(
                [train_xy_arrays[ticker][key] for ticker in sorted(train_xy_arrays.keys())],
                axis=0
            )
            self._test_xy_arrays[key] = np.concatenate(
                [test_xy_arrays[ticker][key] for ticker in sorted(test_xy_arrays.keys())],
                axis=0
            )
        self._train_xy_arrays["N"] = self._train_xy_arrays['x'].shape[0]
        self._test_xy_arrays["N"] = self._test_xy_arrays['x'].shape[0]
        LOG.info(f"  Training has {self._train_xy_arrays['N']} sequences of length {self.model_seq_len}.")
        LOG.info(f"  Testing has {self._test_xy_arrays['N']} sequences of length {self.model_seq_len}.")
        self._test_out = self._test_xy_arrays
    
    def create_train_cv_folds(self):
        LOG.info(f"Making {self.cross_validation_folds} validation folds...")
        train_val_distance = int(np.ceil(self.model_seq_len / self._seq_dist))
        
        train_dates = np.array(sorted(list(set(self._train_xy_arrays['prediction_date'].reshape(-1)))))
        
        fold_size = (len(train_dates) - train_val_distance) // (self.cross_validation_folds + 1)
        
        LOG.info(f"  Generating folds with fold_size={fold_size} and distance between train and validation being {train_val_distance}")
        folds = {}
        for i in range(self.cross_validation_folds):
            train_end_dt = train_dates[fold_size * (i + 1)]
            val_begin_ind = fold_size * (i + 1) + train_val_distance
            val_begin_dt = train_dates[val_begin_ind]
            val_end_dt = train_dates[val_begin_ind + fold_size]
            
            train_inds = np.argwhere(self._train_xy_arrays["prediction_date"] <= train_end_dt)
            val_inds = np.argwhere(
                (val_begin_dt <= self._train_xy_arrays["prediction_date"]) &
                (self._train_xy_arrays["prediction_date"] <= val_end_dt)
            )
            fold_arrs = {
                "train":{
                    "x": self._train_xy_arrays["x"][train_inds],
                    "y": self._train_xy_arrays["y"][train_inds],
                    "prediction_date": self._train_xy_arrays["prediction_date"][train_inds],
                    "ticker": self._train_xy_arrays["ticker"][train_inds],
                },
                "valid":{
                    "x": self._train_xy_arrays["x"][val_inds],
                    "y": self._train_xy_arrays["y"][val_inds],
                    "prediction_date": self._train_xy_arrays["prediction_date"][val_inds],
                    "ticker": self._train_xy_arrays["ticker"][val_inds],
                },
            }
            folds[i] = fold_arrs
            LOG.info(f"    Fold {i} shapes:")
            for sample in fold_arrs:
                LOG.info(f"      {sample} - x: {fold_arrs[sample]['x'].shape}, y: {fold_arrs[sample]['y'].shape}")
        folds["_all_"] = self._train_xy_arrays
        self._train_out = folds
    
    def write_data(self):
        self._save_path = self.output_path.joinpath(f"{len(self._tickers)}tickers")
        LOG.info(f"Saving generated data at {self._save_path.as_posix()}...")
        if not self._save_path.exists():
            LOG.info(f"  Directory doesn't exist, making directory...")
            self._save_path.mkdir(parents=True)
        LOG.info("  Writing train folds...")
        write_pickle_file(self._train_out, self._save_path.joinpath("train.pkl"))
        LOG.info("  Writing test arrays...")
        write_pickle_file(self._test_out, self._save_path.joinpath("test.pkl"))
        
    def load_data(self, path):
        self._save_path = self.output_path.joinpath(path)
        LOG.info(f"Loading generated data from {self._save_path.as_posix()}...")
        if not self._save_path.exists():
            raise FileNotFoundError("Directory doesn't exist, can't load data!")
        LOG.info("  Loading train folds...")
        self._train_out = load_pickle_file(self._save_path.joinpath("train.pkl"))
        LOG.info("  Loading test arrays...")
        self._test_out = load_pickle_file(self._save_path.joinpath("test.pkl"))
        
    def prepare_data(self, tickers=["_all_"]):
        """
        
        tickers: list of str
            list of tickers to generate together; if "_all_", generates with all tickers
            from the given data path
        """
        if tickers[0] == "_all_":
            tickers = [f.stem for f in self.data_path.glob("*.csv")]
        self._tickers = tickers
        self.load_input(tickers)
        self.create_arrays()
        self.create_train_cv_folds()
        self.write_data()