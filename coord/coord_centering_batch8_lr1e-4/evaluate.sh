#!/bin/bash

# 5-fold cross-validation
for i in 0 1 2 3 4
do
    ./split_gold_standard_by_record.R ${i}
    ./prepare_tapping.R
    ./pull_dl_prediction.R ${i} 
    
    # each fold has 5 models
    # remove this because in the raw score there're some missing values
    # for j in 1 2 3 4 5
    # do
    #     ./evaluation.py --input fold_${i}/eva.txt.${j}.test --auc fold_${i}/auc.${j}.txt --auprc fold_${i}/auprc.${j}.txt
    # done

    ./evaluation.py --input feature_test.txt --auc fold_${i}/${i}.individual.auc.txt --auprc fold_${i}/${i}.individual.auprc.txt

    ./evaluation.py --input feature_test_mean.txt --auc fold_${i}/${i}.average.auc.txt --auprc fold_${i}/${i}.average.auprc.txt

    ./evaluation.py --input feature_test_max.txt --auc fold_${i}/${i}.max.auc.txt --auprc fold_${i}/${i}.max.auprc.txt

done
