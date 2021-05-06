#!/usr/bin/env python

import os
import sys
import json
import numpy as np
from glob import iglob

target_dir = "/ssd/gyuanfan/2018/PDDB_tapping/samples/*/*/*"
# txt is for the convenience of checking data 
save_txt = "/ssd/dengkw/tap_data/txt/"
save_npy = "/ssd/dengkw/tap_data/npy/"
target_files = iglob(target_dir)

LOG = open("extract.log", "w")

for f in target_files:
    try:
        with open(f, "r") as FILE:
            data = json.load(FILE)
            data_table = []
            
            for record in data:
                # in order to prevent the same coordinates dropped by set
                # change the brackets type
                data_table.append([record['TapTimeStamp'], record['TappedButtonId']] + eval(record['TapCoordinate'].replace('{', '[').replace('}', ']')))

            data_table = np.array(data_table, dtype=object)
            LOG.write("%s: %s\n" % (f, data_table.shape))

            np.savetxt(save_txt + os.path.dirname(f).split("/")[-1], data_table, delimiter="\t", fmt='%s')
            np.save(save_npy + os.path.dirname(f).split("/")[-1], data_table)
    except:
        print("Fail to process: " + f)
        LOG.write("Fail to process: " + f + "\n")
        print(sys.exc_info()[0])

