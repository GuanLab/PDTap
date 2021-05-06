#!/bin/Rscript

input = scan("stdin", quiet=TRUE)
cat(min(input), max(input), mean(input), median(input), "\n")
