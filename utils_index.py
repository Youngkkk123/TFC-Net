from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import copy


def calcAUC(truth, probability):
    fpr, tpr, thresholds = roc_curve(truth, probability, pos_label=1, drop_intermediate=False)
    thresholds = list(thresholds)
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    best_thresholds = thresholds[maxindex]
    tpr = list(tpr)
    fpr = list(fpr)
    avg_train_auc = auc(fpr, tpr)
    return avg_train_auc, best_thresholds


def calcACCSENSPE(truth, probability, threshold=0.5):
    probability_binary = np.asarray(copy.deepcopy(probability))
    # print
    probability_binary[probability_binary >= threshold] = 1
    probability_binary[probability_binary < threshold] = 0
    probability_binary = probability_binary.tolist()
    confMatrix = confusion_matrix(truth, probability_binary)
    tn = confMatrix[0, 0]
    fp = confMatrix[0, 1]
    tp = confMatrix[1, 1]
    fn = confMatrix[1, 0]
    SEN = tp / (tp + fn)
    SPE = tn / (tn + fp)
    ACC = (tp + tn) / (tp + fp + tn + fn)
    return ACC, SEN, SPE
