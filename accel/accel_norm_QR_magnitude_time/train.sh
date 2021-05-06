#!/bin/bash

#for i in 0,1,2,3,4
for i in 0 1 2 3 4
do
    perl split_gold_standard_by_record.pl ${i}

    perl prepare_tapping_train.pl
    perl prepare_tapping_test.pl
    python train.py tapping_train.txt 2500 guan_version1_2500

    cp fold1_params_50 params
    python predict.py tapping_test.txt 2500 guan_version1_2500
    cp eva.txt eva.txt.1.test
    python predict.py full_tapping_train.txt 2500 guan_version1_2500
    cp eva.txt eva.txt.1.train

    cp fold2_params_50 params
    python predict.py tapping_test.txt 2500 guan_version1_2500
    cp eva.txt eva.txt.2.test
    python predict.py full_tapping_train.txt 2500 guan_version1_2500
    cp eva.txt eva.txt.2.train

    cp fold3_params_50 params
    python predict.py tapping_test.txt 2500 guan_version1_2500
    cp eva.txt eva.txt.3.test
    python predict.py full_tapping_train.txt 2500 guan_version1_2500
    cp eva.txt eva.txt.3.train

    cp fold4_params_50 params
    python predict.py tapping_test.txt 2500 guan_version1_2500
    cp eva.txt eva.txt.4.test
    python predict.py full_tapping_train.txt 2500 guan_version1_2500
    cp eva.txt eva.txt.4.train

    cp fold5_params_50 params
    python predict.py tapping_test.txt 2500 guan_version1_2500
    cp eva.txt eva.txt.5.test
    python predict.py full_tapping_train.txt 2500 guan_version1_2500
    cp eva.txt eva.txt.5.train

    mv fold*params* fold_${i}/
    mv eva.txt* fold_${i}/

done
