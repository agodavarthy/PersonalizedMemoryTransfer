import numpy as np
import javaobj
import os
import pandas as pd 
def load_java_bin(filename):
    """
    Loads a binary serialized file from java
    :param filename: path filenames
    :returns: data loaded
    :rtype: numpy array
    """
    with open(filename, 'rb') as f:
        data = javaobj.load(f)
    return np.array(data.data)
fold = 4
base_directory = "/home/ywang/convmovie/data/gorec_ratings/{}/30/WRMF/".format(fold)
print(base_directory)
item = load_java_bin("%s/itemFactors.bin" % base_directory).astype(np.float32)
user = load_java_bin("%s/userFactors.bin" % base_directory).astype(np.float32)
k = item.shape[1]
save_path = '/home/ywang/convmovie/data/gorec_ratings/{}/{}'.format(fold, k)
if not os.path.exists(save_path):
    os.makedirs(save_path)
print(save_path)
np.savetxt('{}/U.txt'.format(save_path), user)
np.savetxt('{}/V.txt'.format(save_path), item)