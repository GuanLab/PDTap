#!/usr/bin/env python

import argparse
from sklearn.metrics import auc
import numpy as np
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
from scipy.stats import pearsonr, spearmanr

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str, required=True, help="Input file")
ap.add_argument('-a', '--auc', type=str, required=True, help="Output AUC file")
ap.add_argument('-p', '--auprc', type=str, required=True, help="Output AUPRC file")
args = vars(ap.parse_args())

input_file = args['input']
output_auc = args['auc']
output_auprc = args['auprc']

y = np.loadtxt(input_file)
pred = np.loadtxt(input_file)[:,1]

print(y.shape)
fpr, tpr, thresholds = metrics.roc_curve(y[:,0], pred, pos_label=1)
the_auc = metrics.auc(fpr, tpr)
F = open(output_auc, 'w')
F.write('%.4f\n' % the_auc)
F.close()

precision, recall, thresholds = precision_recall_curve(y[:,0], pred)
the_auprc = auc(recall, precision,reorder=False)
F = open(output_auprc,'w')
F.write('%.4f\n' % the_auprc)
F.close()

coef_pearson, pvalue = pearsonr(y[:, 0], pred)
coef_spearman, pvalue = spearmanr(y[:, 0], pred)
F = open("Correlations.txt", "a")
F.write("%s: %.4f\t%.4f\n" % (input_file, coef_pearson, coef_spearman))
F.close()
