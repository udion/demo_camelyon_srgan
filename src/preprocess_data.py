import numpy as np
import os, shutil, sys
import imageio

pre_data_path = '../data/sample_dataset/test/unlabeled/'
post_data_path = '../data/post_dataset/test/unlabeled/'
post_data_indx_path ='../data/post_dataset/'

try:
    os.makedirs(post_data_path)
except OSError:
    pass

all_files = os.listdir(pre_data_path)
total_files = len(all_files)

file_indx_to_name_unique_vals = {}
arr_unique_vals = np.zeros(total_files)
img_buf = []
for idx, fname in enumerate(all_files):
    # print(idx, pre_data_path+fname)
    img = imageio.imread(pre_data_path+fname)
    img_buf.append(img)
    file_indx_to_name_unique_vals[idx] = (fname, len(np.unique(img)))
    arr_unique_vals[idx] = len(np.unique(img))

thresh_u_vals = 200
n = np.sum(arr_unique_vals<thresh_u_vals)
print('number of images not having enough unique vals (to be discared) : ', n)
print('expected number of images to be saved :', total_files-n+1)

count = 0
for i in range(total_files):
    fname, u_val = file_indx_to_name_unique_vals[i][0], file_indx_to_name_unique_vals[i][1]
    if u_val < thresh_u_vals:
        pass
    else:
        img = img_buf[i]
        imageio.imwrite(post_data_path+'{}.png'.format(count), img)
        count += 1




