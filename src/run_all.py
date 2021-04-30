import sys
import glob
import os
from pathlib import Path

if __name__ == "__main__":
    data_path=Path(__file__).absolute().parent.parent.joinpath("data/feature_selected")
    files = glob.glob(os.path.join(data_path, "*.csv"))
    file_names = [os.path.basename(filename).rstrip(".csv") for filename in files]
    for file_name in file_names:
        os.system("python main.py {}".format(file_name))
