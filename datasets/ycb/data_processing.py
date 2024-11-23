import numpy as np

path = '/homes/lt006/code/6D_without_forgetting/datasets/ycb/dataset_config/test_data_list.txt'
import scipy.misc
import scipy.io as scio

list = []
real = []
syn = []
input_file = open(path)
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    if input_line[:5] == 'data/':
        real.append(input_line)
    else:
        syn.append(input_line)
    list.append(input_line)
input_file.close()

list_1 = []
list_2 = []
list_3 = []
list_4 = []
list_5 = []
list_6 = []
list_7 = []
list_8 = []
list_9 = []
list_10 = []
list_11 = []
list_12 = []
list_13 = []
list_14 = []
list_15 = []
list_16 = []
list_17 = []
list_18 = []
list_19 = []
list_20 = []
list_21 = []

for index in range(len(list)):
    # meta = scio.loadmat('/import/smartcameras-002/long/YCB_videos/{1}-meta.mat'.format(list[index]))
    meta = scio.loadmat("/import/smartcameras-002/long/YCB_videos/{}-meta.mat".format(list[index]))
    obj = meta['cls_indexes'].flatten().astype(np.int32)

    if 1 in obj:
        list_1.append(list[index])

    if 2 in obj:
        list_2.append(list[index])

with open("/homes/lt006/code/6D_without_forgetting/datasets/ycb/data_list/test_1.txt", "w") as file:
    for item in list_1:
        file.write((str(item) + "\n"))

with open("/homes/lt006/code/6D_without_forgetting/datasets/ycb/data_list/test_2.txt", "w") as file:
    for item in list_2:
        file.write((str(item) + "\n"))

print("done")

