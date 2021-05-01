import glob
import os
from pathlib import Path
import sys
import time

if __name__ == "__main__":
    """
    python predict_all.py run each of the stock model
    python predict_all.py all run stock with the single all trained model
    """

    data_path = Path(__file__).absolute().parent.joinpath("data/feature_selected")
    files = glob.glob(os.path.join(data_path, "*.csv"))
    file_names = [os.path.basename(filename).rstrip(".csv") for filename in files]
    if len(sys.argv) <= 1:
        for file_name in file_names:
            os.system("python predict.py {}".format(file_name))
    elif sys.argv[1] == "all":
        for file_name in file_names:
            os.system("python predict.py all {}".format(file_name))


