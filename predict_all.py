import glob
import os
from pathlib import Path
import sys
import argparse

parser = argparse.ArgumentParser(
    description="training pipeline"
)
parser.add_argument("-a", help="y or n")
parser.add_argument("-m", help="LSTM or GRU")
# arguments
args = parser.parse_args()

assert args.a in ["y", "n"], "allTicker or not is not specified."
assert args.m in ["LSTM", "GRU"], "model type specified wrong"

if __name__ == "__main__":
    """
    python predict_all.py run each of the stock model
    python predict_all.py all run stock with the single all trained model
    """

    data_path = Path(__file__).absolute().parent.joinpath("data/feature_selected")
    files = glob.glob(os.path.join(data_path, "*.csv"))
    file_names = [os.path.basename(filename).rstrip(".csv") for filename in files]
    for file_name in file_names:
        os.system("python predict.py -t {} -a {} -m {}".format(file_name, args.a, args.m))


