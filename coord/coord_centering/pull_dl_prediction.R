#!/bin/Rscript
library(dplyr)

args = commandArgs(trailingOnly=TRUE)

test_data = read.table("tapping_test.txt", header=F, stringsAsFactors=F, sep="\t")

folder = paste0("./fold_", args[1])
eva_files = list.files(folder, pattern="eva\\.txt\\..+\\.test")
for (FILE in eva_files) {
    data = readLines(file.path(folder, FILE))
    score = as.numeric(sapply(strsplit(data, split="\t"), "[", 2))
    test_data = cbind(test_data, score)
}

colnames(test_data) = c("recordId", "label", "file", "age", "male", "female", "medical.true", "medical.false",
                        "eva1", "eva2", "eva3", "eva4", "eva5")

avg = mean(colMeans(test_data[, 9:13], na.rm=TRUE))

## pull_dl_test
feature_test = data.frame(label=test_data$label,
                          eva=apply(test_data[, 9:13], 1, mean, na.rm=TRUE))
feature_test[is.na(feature_test)] = avg  # make sure the NAs won't appear at the label
write.table(feature_test, "feature_test.txt", sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE)

## pull_dl_individual
tapping_data = read.table("../../Tapping_activity_training.tsv", sep="\t", header=T, stringsAsFactors=F)
merge_data = merge(tapping_data[, c("healthCode", "recordId")], test_data, by="recordId")
merge_data$eva = rowMeans(merge_data[, c("eva1", "eva2", "eva3", "eva4", "eva5")], na.rm=TRUE)
merge_data$eva[is.na(merge_data$eva)] = avg

# max
# Make sure that one healthcode only has one label
feature_test_max = merge_data %>% group_by(healthCode, label) %>%
    summarise(maxEva = max(eva, na.rm=T))
feature_test_max = feature_test_max[, 2:3]
write.table(feature_test_max, "feature_test_max.txt", sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE)

# mean
# Make sure that one healthcode only has one label
feature_test_mean = merge_data %>% group_by(healthCode, label) %>%
    summarise(meanEva = mean(eva, na.rm=T))
feature_test_mean = feature_test_mean[, 2:3]
write.table(feature_test_mean, "feature_test_mean.txt", sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE)










