import sys
import glob
import os
from pathlib import Path
import time
import argparse

parser = argparse.ArgumentParser(
    description="prediction pipeline"
)
parser.add_argument("-m", help="model type: LSTM or GRU")
args = parser.parse_args()
assert args.m in ["LSTM", "GRU"], "model type specified wrong"

if __name__ == "__main__":
    model_type = args.m
    data_path = Path(__file__).absolute().parent.joinpath("data/feature_selected")
    files = glob.glob(os.path.join(data_path, "*.csv"))
    file_names = [os.path.basename(filename).rstrip(".csv") for filename in files]
    for file_name in file_names:
        os.system("python main.py -t {} -m {}".format(file_name, model_type))
