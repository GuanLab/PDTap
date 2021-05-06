#!/bin/bash


cat Correlations.txt | awk -F " " '{if ($1=="feature_test.txt:") print $2, $3}' | cut -d " " -f 1 > datafile_pearson
cat Correlations.txt | awk -F " " '{if ($1=="feature_test.txt:") print $2, $3}' | cut -d " " -f 2 > datafile_spearman

cat Correlations.txt | awk -F " " '{if ($1=="feature_test_mean.txt:") print $2, $3}' | cut -d " " -f 1 > mean_datafile_pearson
cat Correlations.txt | awk -F " " '{if ($1=="feature_test_mean.txt:") print $2, $3}' | cut -d " " -f 2 > mean_datafile_spearman

cat Correlations.txt | awk -F " " '{if ($1=="feature_test_max.txt:") print $2, $3}' | cut -d " " -f 1 > max_datafile_pearson
cat Correlations.txt | awk -F " " '{if ($1=="feature_test_max.txt:") print $2, $3}' | cut -d " " -f 2 > max_datafile_spearman

