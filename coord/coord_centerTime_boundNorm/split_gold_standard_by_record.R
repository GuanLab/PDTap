#!/bin/Rscript
library(dplyr)
library(data.table)
library(mltools)

args = commandArgs(trailingOnly=TRUE)
set.seed(as.integer(args))

## demographic and file information of raw data
demographic = read.table('/ssd/dengkw/Demographic.tsv', sep="\t", header=T, stringsAsFactors=F)
tapping = read.table('/ssd/dengkw/Tapping_activity_training.tsv', sep="\t", header=T, stringsAsFactors=F)

demographic = demographic[, c('healthCode', 'professional.diagnosis', 'age', 'gender', 'medical.usage.yesterday')]
demographic = demographic[demographic$professional.diagnosis != "", ]
demographic$professional.diagnosis = ifelse(demographic$professional.diagnosis == "true", 1, 0)


## fill the NA in age
demo_summary = demographic %>% filter(!is.na(gender) & !is.na(medical.usage.yesterday)) %>% 
    group_by(gender, medical.usage.yesterday, professional.diagnosis) %>% 
    summarise(mean.age = mean(age, na.rm = T))
demo_summary_diag = demographic %>% group_by(professional.diagnosis) %>% 
    summarise(mean.age = mean(age, na.rm = T))

# if the corresponding gender and medical information are not missing, use the average age grouped by these variables and diagnosis
# else just use the average age group by diagnosis
for (i in which(is.na(demographic$age))) {
    gender = demographic[i, 'gender']
    medical = demographic[i, 'medical.usage.yesterday']
    diagnosis = demographic[i, 'professional.diagnosis']
    if (!is.na(gender) & !is.na(medical)) {
        demographic[i, 'age'] = demo_summary[which(demo_summary$gender == gender & demo_summary$medical.usage.yesterday == medical & demo_summary$professional.diagnosis == diagnosis), 'mean.age']
    } else {
        demographic[i, 'age'] = demo_summary_diag[which(demo_summary_diag$professional.diagnosis == diagnosis), 'mean.age']
    }
}



## one-hot code gender and medical usage yesterday
# 1. convert to factor
demographic$gender = ifelse(demographic$gender %in% c("Male", "Female"), demographic$gender, NA)
demographic$gender = as.factor(demographic$gender)
levels(demographic$gender) = c('Male', 'Female')

demographic$medical.usage.yesterday = ifelse(demographic$medical.usage.yesterday %in% c("true", "false"), 
                                             demographic$medical.usage.yesterday, NA)
demographic$medical.usage.yesterday = as.factor(demographic$medical.usage.yesterday)
levels(demographic$medical.usage.yesterday) = c('true', 'false')

# 2. convert to data.table
demographic = as.data.table(demographic)
demographic = one_hot(demographic, cols=c('gender', 'medical.usage.yesterday'), sparsifyNA=F)

# 3. fill the NAs with 0.5
demographic[is.na(demographic)] = 0.5
# colnames(demographic)



## split the train/test test
## 75% train and 25% test
split_random = runif(nrow(demographic), min=0, max=1)

hcode_train = demographic[split_random < 0.75, ]
hcode_test = demographic[split_random >= 0.75, ]




## use the health code to map the record ids in tapping data
train = merge(tapping, hcode_train, by="healthCode")
train = train[, c('recordId', 'professional.diagnosis', 'tapping_results.json.TappingSamples', 
                  'tapping_results.json.ButtonRectLeft', 'tapping_results.json.ButtonRectRight',
                  'age', 'gender_Male', 'gender_Female',
                  'medical.usage.yesterday_true', 'medical.usage.yesterday_false')]
test = merge(tapping, hcode_test, by="healthCode")
test = test[, c('recordId', 'professional.diagnosis', 'tapping_results.json.TappingSamples', 
                'tapping_results.json.ButtonRectLeft', 'tapping_results.json.ButtonRectRight',
                'age', 'gender_Male', 'gender_Female',
                'medical.usage.yesterday_true', 'medical.usage.yesterday_false')]
table(train$professional.diagnosis)

## write to file
write.table(x=train, file='train_gs_r.dat', sep='\t', row.names=F, col.names=F, quote=F)
write.table(x=test, file='test_gs_r.dat', sep='\t', row.names=F, col.names=F, quote=F)
