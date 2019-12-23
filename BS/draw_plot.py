import sys
import os

import sklearn.metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


result_dir = "./test_result"


def main():
    models = sys.argv[1:]
    # models = os.listdir("./test_result")
    # models = [each[:-6] for each in models if each.endswith("npy")]
    for model in models:
        x = np.load(os.path.join(result_dir, model + "_x.npy"))  # best p
        y = np.load(os.path.join(result_dir, model + "_y.npy"))  # best r
        f1 = (2 * x * y / (x + y + 1e-20)).max()
        auc = sklearn.metrics.auc(x=x, y=y)
        plt.plot(x, y, lw=2, label=model)
        print(model + " : " + "auc = " + str(auc) + " | " + "max F1 = " + str(f1) +
              "    P@100: {0} | P@200: {1} | P@300: {2} | Mean: {3} | "
              "P@R0.1: {4} | P@R0.2: {5} | P@R0.3: {6} | P@R0.4: {7})".format(
                  y[100], y[200], y[300], (y[100] + y[200] + y[300]) / 3,
                  y[int(len(y)*0.1)], y[int(len(y)*0.2)], y[int(len(y)*0.3)], y[int(len(y)*0.4)]
                )
              )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0.3, 1.0)
    plt.xlim(0.0, 0.4)
    plt.title("Precision-Recall")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, "pr_curve"))


if __name__ == "__main__":
    main()
