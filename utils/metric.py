import os
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_EER(scores, labels):
    if len(scores) != len(labels):
        raise Exception('length between scores and labels is different')
    elif len(scores) == 0:
        raise Exception("There's no elements in scores")
        
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return EER

def calculate_MinDCF(scores, labels, p_target=0.01, c_miss=10, c_false_alarm=1):
    if len(scores) != len(labels):
        raise Exception('length between scores and labels is different')
    elif len(scores) == 0:
        raise Exception("There's no elements in scores")
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    dcf = c_miss * fnr * p_target + c_false_alarm * fpr * (1 - p_target)
    c_det = np.min(dcf)
    c_def = min(c_miss * p_target, c_false_alarm * (1 - p_target))
    return c_det / c_def

def draw_histogram(scores, labels):
    positive = []
    negative = []
    for i in range(len(labels)):
        if int(labels[i]) == 1:
            positive.append(scores[i])
        else:
            negative.append(scores[i])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.ylim([0, 500])
    ax.hist(positive, label="target", bins=200, color="blue", alpha=0.5)
    ax.hist(negative, label="nontarget", bins=200, color="red", alpha=0.5)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    plt.ylabel("#trial")
    plt.legend(loc="best")

    if os.path.exists('histogram.png'):
        os.remove('histogram.png')    

    plt.savefig("histogram.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    img = Image.open("histogram.png")
    
    return img

def draw_heatmap(title, x_axis, y_axis, featmap):
    plt.matshow(featmap, cmap=plt.get_cmap('GnBu'), aspect='auto')
    plt.colorbar(shrink=0.8, aspect=10)
    plt.clim(np.min(featmap), np.max(featmap))
    plt.title(title, fontsize=20)
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    plt.savefig("heatmap.png", dpi=400, bbox_inches="tight")
    plt.close()
    img = Image.open("heatmap.png")
    os.remove('heatmap.png')

    return img