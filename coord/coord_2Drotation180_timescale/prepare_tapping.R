#!/bin/Rscript

path_generator = function(file_string) {
    basepath = "/ssd/dengkw/tap_data/npy/"
    path = paste0(basepath, file_string, ".npy")
    if (file.exists(path)) {
        return(path)
    } else {
        return(NA)
    }
}


file_generator = function(data_type) {
    cur_file = paste0(data_type, "_gs_r.dat")
    data_table = read.table(cur_file, sep="\t", header=F, na.strings="")
    colnames(data_table) = c("recordId", "prognosis", "accel_tapping", "age", "Male", "Female", "medical.true", "medical.false")

    data_table$accel_tapping = unlist(lapply(data_table[["accel_tapping"]], path_generator))

    output_file = paste0("tapping_", data_type, ".txt")
    write.table(na.omit(data_table), file=output_file, sep="\t", row.names=F, col.names=F, quote=F)
}

file_generator("train")
file_generator("test")
