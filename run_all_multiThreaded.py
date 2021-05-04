import glob
import os
from pathlib import Path
import threading


def train_model(file_name):
    """

    :param file_name: input csv file name
    :return:
    """
    os.system("python main.py {}".format(file_name))


if __name__ == "__main__":
    data_path = Path(__file__).absolute().parent.parent.joinpath("data/feature_selected")
    files = glob.glob(os.path.join(data_path, "*.csv"))
    file_names = [os.path.basename(filename).rstrip(".csv") for filename in files]
    # number of threads to use
    N = 8
    # stock id
    id_ = 0
    while id_ < len(file_names):
        threads = []
        for i in range(N):
            t = threading.Thread(target=train_model, args=(file_names[id_],))
            threads.append(t)
            id_ += 1
            if id_ >= len(file_names):
                break
        # execute functions with N threads
        # start threads
        for t in threads:
            t.start()
        # join threads
        for t in threads:
            t.join()
