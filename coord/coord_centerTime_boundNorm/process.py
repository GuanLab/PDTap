import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def train_validate_split(input_file, seed, test_size=0.5):
    # Xs are the file names of the raw data and other demographic data
    # ys are the labels
    Xs, ys = [], []
    
    with open(input_file, 'r') as DATA:
        for line in DATA:
            data_list = line.strip().split("\t")
            y, X = data_list[1], data_list[2:]
            
            # Balance sample #0 < #1
            if y == 0 or y == "0":
                Xs.append(X)
                ys.append(y)
                
            Xs.append(X)
            ys.append(y)
    
    X_train, X_validate, y_train, y_validate = train_test_split(Xs, ys, test_size=test_size, random_state=seed, shuffle=True)
    
    path_train, bound_train = np.asarray(X_train)[:, 0], np.asarray(X_train)[:, 1:3]
    path_validate, bound_validate = np.asarray(X_validate)[:, 0], np.asarray(X_validate)[:, 1:3]

    path_train = path_train.astype("str")
    path_validate = path_validate.astype("str")
    bound_train = bound_train.astype("str")
    bound_validate = bound_validate.astype("str")

    return path_train, bound_train, path_validate, bound_validate, np.asarray(y_train, dtype="int32"), np.asarray(y_validate, dtype="int32")


def normalization(accel_data):
    # Channel at the back
    # shape: (length, 3)
    _mean = np.mean(accel_data, axis=0)
    _sd = np.std(accel_data, axis=0)
    return (accel_data - _mean) / _sd


def centering(accel_data):
    # Channel at the back
    # shape: (length, 3)
    _mean = np.mean(accel_data, axis=0)
    return accel_data - _mean


def fix_size(accel_data, size=2500):
    # Channel in the front
    # shape: (3, length)
    if size >= len(accel_data):
        tmp = np.zeros((accel_data.shape[0], size))
        tmp[:, 0:accel_data.shape[1]] = accel_data
        return tmp
    else:
        return accel_data[:, 0:size]


def scale_time(accel_data, scale):
    nrow, ncol = accel_data.shape  # 3, 2500
    ncol_new = int(ncol * scale)
    scaled_data = cv2.resize(accel_data, (ncol_new, nrow))
    if ncol_new >= ncol:  # scale >= 1
        return scaled_data[:, 0:ncol]
    else:
        tmp = np.zeros((nrow, ncol))
        tmp[:, 0:ncol_new] = scaled_data
        return tmp


def scale_magnitude(accel_data, scales):
    assert accel_data.shape[0] == len(scales), "Each channel should has its own scaling factor"
    for i in range(accel_data.shape[0]):
        accel_data[i, :] = accel_data[i, :] * scales[i]
    return


def quarternion_rotation(accel_data, theta, x, y, z):
    # A new coordinate system
    axis = np.array([x, y, z])
    axis = axis / np.sqrt(np.dot(axis, axis))

    # generate the quarternion rotation matrix
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a ** 2, b ** 2, c ** 2, d ** 2
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    qrMatrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)], \
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)], \
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    return np.dot(qrMatrix, accel_data)


def bound_parser(bound):

    # input format: "{{x1, y1}, {x2, y2}}"
    # only choose the first set because the second is always {104, 104}

    return eval(bound.replace("{", "(").replace("}", ")"))[0]


def bound_normalization(coords, bounds):

    # coords input format: [[buttonId, x, y], ...]
    # bounds input format: ["{{xl, yl}, {104, 104}}", "{{xr, yr}, {104, 104}}"]

    xl, yl = bound_parser(bounds[0])
    xr, yr = bound_parser(bounds[1])
    bound_map = {"TappedButtonLeft": (xl, yl), "TappedButtonRight": (xr, yr)}

    for i, record in enumerate(coords):

        if record[0] == "TappedButtonNone":
            # actually, don't need to comapre the record, just compare the x-val with the bound
            if record[1] < xr:
                record[1] -= bound_map["TappedButtonLeft"][0]
                record[2] -= bound_map["TappedButtonLeft"][1]
            else:
                record[1] -= bound_map["TappedButtonRight"][0]
                record[2] -= bound_map["TappedButtonRight"][1]

        else:
            record[1] -= bound_map[record[0]][0]
            record[2] -= bound_map[record[0]][1]
    



