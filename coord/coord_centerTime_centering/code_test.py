from process import *

input_file = "tapping_train.txt"
path_train, demo_train, path_validate, demo_validate, label_train, label_validate = train_validate_split(input_file, 0, 0.5)
print(path_train)
print(path_train.shape[0])
print(demo_train)
print(label_train)
