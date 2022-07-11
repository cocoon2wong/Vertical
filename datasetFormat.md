<!--
 * @Author: Conghao Wong
 * @Date: 2022-07-11 21:25:30
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2022-07-11 21:30:28
 * @Description: file content
 * @Github: https://github.com/cocoon2wong
 * Copyright 2022 Conghao Wong, All Rights Reserved.
-->

# Dataset Format for Training `V^2-Net`

Before training `V^2-Net` on your own dataset, you should add your dataset information to the `datasets` directory.
A dataset contains a `dataset splits file` and several `sub-dataset` files.
For example, we have added the ETH-UCY and SDD dataset files in the `datasets` folder:

```none
datasets
|___eth.plist
|___hotel.plist
|___sdd.plist
|___univ.plist
|___zara1.plist
|___zara2.plist
|___subsets
    |___...
```

## Dataset Splits File

It contains the dataset splits used for training and evaluation.
For example, you can save the following python `dict` object as the `MyDataset.plist` (Maybe a python package like `biplist` is needed):

```python
my_dataset = {
'test': ['test_subset1'],
'train': ['train_subset1', 'train_subset2', 'train_subset3'],
'val': ['val_subset1', 'val_subset2'],
}
```

## Sub-Dataset File

You should edit and put information about all your sub-dataset that you have written into the dataset splits file into the `/datasets/subsets` directory.
For example, you can save the following python `dict` object as the `test_subset1.plist`:

```python
test_subset1 = {
'dataset': 'test_subset1',    # name of that sub-dataset
'dataset_dir': '....',        # root dir for your dataset csv file
'order': [1, 0],              # x-y order in your csv file
'paras': [1, 30],             # [your data fps, your video fps]
'scale': 1,                   # scale when save visualization figs
'video_path': '....',         # path for the corresponding video file 
}
```

Besides, all trajectories should be saved in the following `true_pos_.csv` format:

- Size of the matrix is 4 x numTrajectoryPoints
- The first row contains all the frame numbers
- The second row contains all the pedestrian IDs
- The third row contains all the y-coordinates
- The fourth row contains all the x-coordinates
