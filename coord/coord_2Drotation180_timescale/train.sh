#!/bin/bash
CUDA_VISIBLE_DEVICES=1
#for i in 0,1,2,3,4
for i in 0 1 2 3 4 
do
    ./split_gold_standard_by_record.R ${i}
    ./prepare_tapping.R
   
    ./train.py --train tapping_train.txt --size 800 --model guan_version1_800 --fold ${i}

    ./predict.py --test tapping_test.txt --size 800 --model guan_version1_800 --params model1_params_50 --output eva.txt.1.test

    ./predict.py --test tapping_test.txt --size 800 --model guan_version1_800 --params model2_params_50 --output eva.txt.2.test

    ./predict.py --test tapping_test.txt --size 800 --model guan_version1_800 --params model3_params_50 --output eva.txt.3.test
    
    ./predict.py --test tapping_test.txt --size 800 --model guan_version1_800 --params model4_params_50 --output eva.txt.4.test

    ./predict.py --test tapping_test.txt --size 800 --model guan_version1_800 --params model5_params_50 --output eva.txt.5.test

    mv model*params* fold_${i}/
    mv eva.txt* fold_${i}/

done
