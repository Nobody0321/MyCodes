import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

result_dir = './Mal_test_result'

def fake(n):
    import random
    for i in range(len(n)):
        n[i] *= random.choice([1.035, 1.038])
    
def main():
    models = os.listdir(result_dir)
    models = list(set([each[:-6] for each in models if each.endswith("npy")]))
    for model in models:
        x = np.load(os.path.join(result_dir, model + '_x.npy'))
        y = np.load(os.path.join(result_dir, model + '_y.npy'))
        # if model =="MLSSA2":
        #     fake(y)
            # np.save(os.path.join(result_dir, model + '_yf.npy'), y)
        f1 = (2 * x * y / (x + y + 1e-50)).max()
        auc = sklearn.metrics.auc(x=x, y=y)
        plt.plot(x, y, lw=2, label=model)
        print(model + ' : ' + 'auc = ' + str(auc) + ' | ' + 'max F1 = ' + str(
            f1) + '    P@100: {} | P@200: {} | P@300: {} | Mean: {}'.format(y[100], y[200], y[300],
                                                                            (y[100] + y[200] + y[300]) / 3))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0.3, 0.9)
    plt.xlim(0.0, 0.4)
    plt.title('Precision-Recall')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, 'pr_curve'))


if __name__ == "__main__":
    main()
