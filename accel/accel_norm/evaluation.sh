#!/bin/bash

#for i in 0,1,2,3,4
for i in 0 1 2 3 4
do
    perl split_gold_standard_by_record.pl ${i}


    perl pull_dl_prediction_test.pl ${i}
    python evaluation.py
    cp auc.txt ${i}.individual.auc.txt

    perl pull_dl_prediction_test_by_individual_mean.pl ${i}
    python evaluation.py
    cp auc.txt ${i}.average.auc.txt

    perl pull_dl_prediction_test_by_individual_max.pl ${i}
    python evaluation.py
    cp auc.txt ${i}.max.auc.txt
    mv  *auc* fold_${i}/
    mv  *auprc* fold_${i}/

done
