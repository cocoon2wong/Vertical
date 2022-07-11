'''
Author: Conghao Wong
Date: 2020-11-09 21:29:41
LastEditors: Conghao Wong
LastEditTime: 2020-11-26 21:23:20
Description: file content
'''

import numpy as np
from tqdm import tqdm

sdd_sets = {
    'quad'          :   [[0, 1, 2, 3], 100.0],
    'little'        :   [[0, 1, 2, 3], 100.0],
    'deathCircle'   :   [[0, 1, 2, 3, 4], 100.0],
    'hyang'         :   [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 100.0],
    'nexus'         :   [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 100.0],
    'coupa'         :   [[0, 1, 2, 3], 100.0],
    'bookstore'     :   [[0, 1, 2, 3, 4, 5, 6], 100.0],
    'gates'         :   [[0, 1, 2, 3, 4, 5, 6, 7, 8], 100.0],
}

for name in sdd_sets:
    for video_id in sdd_sets[name][0]:
        scale = sdd_sets[name][1]

        target_txt = './data/sdd/{}/video{}/annotations.txt'.format(name, video_id)
        save_format = './data/sdd/{}/video{}/true_pos_.csv'.format(name, video_id)

        # data = []
        csv_data = []
        with open(target_txt, 'r') as f:
            while True:
                data_original = f.readline()

                if data_original:
                    data_original = data_original.split(' ')
                    if data_original[6] == '1':
                        continue

                    csv_data_c = [
                        float(data_original[5]),
                        float(data_original[0]),
                        (float(data_original[1]) + float(data_original[3]))/(2*scale),
                        (float(data_original[2]) + float(data_original[4]))/(2*scale),
                    ]
                    csv_data.append(csv_data_c)
                    
                else:
                    break

        print('{} Done.'.format(save_format))
        csv_data = np.array(csv_data)
        np.savetxt(save_format, csv_data.T, delimiter=',')