<!--
 * @Author: Conghao Wong
 * @Date: 2021-08-05 15:51:15
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2022-07-11 21:26:48
 * @Description: file content
 * @Github: https://github.com/cocoon2wong
 * Copyright 2022 Conghao Wong, All Rights Reserved.
-->

# Codes for View Vertically: A Hierarchical Network for Trajectory Prediction via Fourier Spectrums

![status](https://img.shields.io/github/workflow/status/cocoon2wong/Project-Jocker/Python%20application/Vertical)
![$V^2$-Net](./vmethod.png)

## Abstract

Understanding and forecasting future trajectories of agents are critical for behavior analysis, robot navigation, autonomous cars, and other related applications.
Previous methods mostly treat trajectory prediction as time sequence generation.
Different from them, this work studies agents' trajectories in a "vertical" view, i.e., modeling and forecasting trajectories from the spectral domain.
Different frequency bands in the trajectory spectrums could hierarchically reflect agents' motion preferences at different scales.
The low-frequency and high-frequency portions could represent their coarse motion trends and fine motion variations, respectively.
Accordingly, we propose a hierarchical network V$^2$-Net, which contains two sub-networks, to hierarchically model and predict agents' trajectories with trajectory spectrums.
The coarse-level keypoints estimation sub-network first predicts the "minimal" spectrums of agents' trajectories on several "key" frequency portions.
Then the fine-level spectrum interpolation sub-network interpolates the spectrums to reconstruct the final predictions.
Experimental results display the competitiveness and superiority of V$^2$-Net on both ETH-UCY benchmark and the Stanford Drone Dataset.

## Citation

If you find this work useful, it would be grateful to cite our paper!

```bib
@article{wong2021view,
  title={View Vertically: A hierarchical network for trajectory prediction via fourier spectrums},
  author={Wong, Conghao and Xia, Beihao and Hong, Ziming and Peng, Qinmu and Yuan, Wei and Cao, Qiong and Yang, Yibo and You, Xinge},
  journal={arXiv preprint arXiv:2110.07288},
  year={2021}
}
```

## Requirements

The codes are developed with python 3.9.
Additional packages used are included in the `requirements.txt` file.
We recommend installing the above versions of the python packages in a virtual environment (like the `conda` environment), otherwise there *COULD* be other problems due to the package version conflicts.

Run the following command to install the required packages in your python  environment:

```bash
pip install -r requirements.txt
```

## Training On Your Datasets

The `V^2-Net` contains two main sub-networks, the coarse-level keypoints estimation sub-network, and the fine-level spectrum interpolation sub-network.
`V^2-Net` forecast agents' multiple trajectories end-to-end.
Considering that most of the loss function terms used to optimize the model work within one sub-network alone, we divide `V^2-Net` into `V^2-Net-a` and `V^2-Net-b`, and apply gradient descent separately for easier training.
You can train your own `V^2-Net` weights on your datasets by training each of these two sub-networks.
After training, you can still use it as a regular end-to-end model.

### Dataset

Before training `V^2-Net` on your own dataset, you should add your dataset information to the `datasets` directory.
See [this document](./datasetFormat.md) for details.

### `V^2-Net-a`

It is the coarse-level keypoints estimation sub-network.
To train the `V^2-Net-a`, you can pass the `--model va` argument to run the `main.py`.
You should also specify the temporal keypoint indexes in the predicted period.
For example, when you want to train a model that predicts future 12 frames of trajectories, and you would like to set $N_{key} = 3$ (which is the same as the basic settings in our paper), you can pass the `--key_points 3_7_11` argument when training.
Please note that indexes start with `0`.
You can also try any other keypoints settings or combinations to train and obtain the `V^2-Net-a` that best fits your datasets.
Please refer to section `Args Used` to learn how other args work when training and evaluating.
Note that do not pass any value to `--load` when training, or it will start *evaluating* the loaded model.

For a quick start, you can train the `V^2-Net-a` via the following minimum arguments:

```bash
python main.py --model va --key_points 3_7_11 --test_set MyDataset
```

### `V^2-Net-b`

It is the fine-level spectrum interpolation sub-network.
You can pass the `--model vb` to run the training.
Please note that you should specify the number of temporal keypoints.
For example, you can pass the `--points 3` to train the corresponding sub-network that takes 3 temporal keypoints or their spectrums as the input.
Similar to the above `V^2-Net-a`, you can train the `V^2-Net-b` with the following minimum arguments:

```bash
python main.py --model vb --points 3 --test_set MyDataset
```

## Evaluation

You can use the following command to evaluate the `V^2-Net` performance end-to-end:

```bash
python main.py \
  --model V \
  --loada A_MODEL_PATH \
  --loadb B_MODEL_PATH
```

Where `A_MODEL_PATH` and `B_MODEL_PATH` are the folders of the two sub-networks' weights.

## Pre-Trained Models

We have provided our pre-trained model weights to help you quickly evaluate the `V^2-Net` performance.
We have uploaded our model weights in the `weights` folder.
It contains model weights trained on `ETH-UCY` by the `leave-one-out` stragety, and model weights trained on `SDD` via the dataset split method from [SimAug](https://github.com/JunweiLiang/Multiverse).

```null
REPO_ROOT_DIR
  - weights
    - vertical
      - a_eth
      - a_hotel
      - a_sdd
      - a_univ
      - a_zara1
      - a_zara2
      - b_universal
```

You can start the quick evaluation via the following commands:

```bash
for dataset in eth hotel univ zara1 zara2 sdd
  python main.py \
    --model V \
    --loada ./weights/vertical/a_${dataset} \
    --loadb ./weights/vertical/b_universal
```

You can also start testing the fast version of `V^2-Net` by passing the argument `--loadb l` like:

```bash
for dataset in eth hotel univ zara1 zara2 sdd
  python main.py \
    --model V \
    --loada ./weights/vertical/a_${dataset} \
    --loadb l
```

We have prepared model outputs that work correctly on the zara1 dataset, details of which can be found in https://github.com/cocoon2wong/Vertical/actions.

If you have the dataset videos and put them into the `videos` folder, you can draw the visualized results by adding the `--draw_reuslts 1` argument.
If you want to draw visualized trajectories like what our paper shows, you can add the additional `--draw_distribution 2` argument:

![Visualization](./fig_vis.png)

## Evaluation of the Usage of Spectrums

We design the minimal vertical model to directly evaluate the metrics improvements brought by the usage of DFT (i.e., the trajectory spectrums).
The minimal V model considers nothing except agents' observed trajectories when forecasting.
You can start a quick training to see how the DFT helps improve the prediction accuracy by changing the argument `--T` between `[none, fft]` via the following scripts:

```bash
for ds in eth hotel univ zara1 zara2
  for T in none fft
    python main.py \
      --model mv \
      --test_set ${ds} \
      --T ${T}
```

You can also [download](drive.google.com)(⚠️NOT UPLOAD YET) and unzip our weights into the `weights/vertical_minimal` folder, then run the following test scripts:

```bash
for name in FFTmv mv
  for ds in eth hotel univ zara1 zara2
    python main.py --load ./weights/vertical_minimal/${name}${ds}
```

Test results will be saved in the `test.log` file.
You can find the following results if everything runs correctly:

```log
[2022-07-06 10:28:59,536][INFO] `MinimalV`: ./weights/vertical_minimal/FFTmveth, eth, {'ADE(m)': 0.79980284, 'FDE(m)': 1.5165437}
[2022-07-06 10:29:02,438][INFO] `MinimalV`: ./weights/vertical_minimal/FFTmvhotel, hotel, {'ADE(m)': 0.22864725, 'FDE(m)': 0.38144386}
[2022-07-06 10:29:15,459][INFO] `MinimalV`: ./weights/vertical_minimal/FFTmvuniv, univ, {'ADE(m)': 0.559813, 'FDE(m)': 1.1061481}
[2022-07-06 10:29:19,675][INFO] `MinimalV`: ./weights/vertical_minimal/FFTmvzara1, zara1, {'ADE(m)': 0.45233154, 'FDE(m)': 0.9287788}
[2022-07-06 10:29:25,595][INFO] `MinimalV`: ./weights/vertical_minimal/FFTmvzara2, zara2, {'ADE(m)': 0.34826145, 'FDE(m)': 0.71161735}
[2022-07-06 10:29:29,694][INFO] `MinimalV`: ./weights/vertical_minimal/mveth, eth, {'ADE(m)': 0.83624077, 'FDE(m)': 1.666721}
[2022-07-06 10:29:32,632][INFO] `MinimalV`: ./weights/vertical_minimal/mvhotel, hotel, {'ADE(m)': 0.2543166, 'FDE(m)': 0.4409294}
[2022-07-06 10:29:45,396][INFO] `MinimalV`: ./weights/vertical_minimal/mvuniv, univ, {'ADE(m)': 0.7743274, 'FDE(m)': 1.3987076}
[2022-07-06 10:29:49,126][INFO] `MinimalV`: ./weights/vertical_minimal/mvzara1, zara1, {'ADE(m)': 0.48137394, 'FDE(m)': 0.97067535}
[2022-07-06 10:29:54,872][INFO] `MinimalV`: ./weights/vertical_minimal/mvzara2, zara2, {'ADE(m)': 0.38129684, 'FDE(m)': 0.7475274}
```

You can find the considerable ADE and FDE improvements brought by the DFT (or called the trajectory spectrums) in the above logs.
Please note that the prediction performance is quite bad due to the simple structure of the *minimal* model, and it considers nothing about agents' interactions and multimodality.

## Args Used

Please specific your customized args when training or testing your model through the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 --ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.
All args and their usages when training and testing are listed below.
Args with `argtype='static'` means that their values can not be changed once after training.

<!-- DO NOT CHANGE THIS LINE -->
### Basic args

- `--K_train`, type=`int`, argtype=`'static'`.
  Number of multiple generations when training. This arg only works for `Generative Models`.
  The default value is `10`.
- `--K`, type=`int`, argtype=`'dynamic'`.
  Number of multiple generations when test. This arg only works for `Generative Models`.
  The default value is `20`.
- `--batch_size`, type=`int`, argtype=`'dynamic'`.
  Batch size when implementation.
  The default value is `5000`.
- `--draw_distribution`, type=`int`, argtype=`'dynamic'`.
  Conrtols if draw distributions of predictions instead of points.
  The default value is `0`.
- `--draw_results`, type=`int`, argtype=`'dynamic'`.
  Controls if draw visualized results on video frames. Make sure that you have put video files into `./videos` according to the specific name way.
  The default value is `0`.
- `--epochs`, type=`int`, argtype=`'static'`.
  Maximum training epochs.
  The default value is `500`.
- `--force_set`, type=`str`, argtype=`'dynamic'`.
  Force test dataset. Only works when evaluating when `test_mode` is `one`.
  The default value is `'null'`.
- `--gpu`, type=`str`, argtype=`'dynamic'`.
  Speed up training or test if you have at least one nvidia GPU. If you have no GPUs or want to run the code on your CPU, please set it to `-1`.
  The default value is `'0'`.
- `--load`, type=`str`, argtype=`'dynamic'`.
  Folder to load model. If set to `null`, it will start training new models according to other args.
  The default value is `'null'`.
- `--log_dir`, type=`str`, argtype=`'static'`.
  Folder to save training logs and models. If set to `null`, logs will save at `args.save_base_dir/current_model`.
  The default value is `dir_check(default_log_dir)`.
- `--lr`, type=`float`, argtype=`'static'`.
  Learning rate.
  The default value is `0.001`.
- `--model_name`, type=`str`, argtype=`'static'`.
  Customized model name.
  The default value is `'model'`.
- `--model`, type=`str`, argtype=`'static'`.
  Model type used to train or test.
  The default value is `'none'`.
- `--obs_frames`, type=`int`, argtype=`'static'`.
  Observation frames for prediction.
  The default value is `8`.
- `--pred_frames`, type=`int`, argtype=`'static'`.
  Prediction frames.
  The default value is `12`.
- `--restore`, type=`str`, argtype=`'dynamic'`.
  Path to restore the pre-trained weights before training. It will not restore any weights if `args.restore == 'null'`.
  The default value is `'null'`.
- `--save_base_dir`, type=`str`, argtype=`'static'`.
  Base folder to save all running logs.
  The default value is `'./logs'`.
- `--save_model`, type=`int`, argtype=`'static'`.
  Controls if save the final model at the end of training.
  The default value is `1`.
- `--start_test_percent`, type=`float`, argtype=`'static'`.
  Set when to start validation during training. Range of this arg is `0 <= x <= 1`. Validation will start at `epoch = args.epochs * args.start_test_percent`.
  The default value is `0.0`.
- `--step`, type=`int`, argtype=`'dynamic'`.
  Frame interval for sampling training data.
  The default value is `1`.
- `--test_mode`, type=`str`, argtype=`'dynamic'`.
  Test settings, canbe `'one'` or `'all'` or `'mix'`. When set it to `one`, it will test the model on the `args.force_set` only; When set it to `all`, it will test on each of the test dataset in `args.test_set`; When set it to `mix`, it will test on all test dataset in `args.test_set` together.
  The default value is `'mix'`.
- `--test_set`, type=`str`, argtype=`'static'`.
  Dataset used when training or evaluating.
  The default value is `'zara1'`.
- `--test_step`, type=`int`, argtype=`'static'`.
  Epoch interval to run validation during training. """ return self._get('test_step', 3, argtype='static') """ Trajectory Prediction Args 
  The default value is `3`.
- `--use_extra_maps`, type=`int`, argtype=`'dynamic'`.
  Controls if uses the calculated trajectory maps or the given trajectory maps. The model will load maps from `./dataset_npz/.../agent1_maps/trajMap.png` if set it to `0`, and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` if set this argument to `1`.
  The default value is `0`.
- `--use_maps`, type=`int`, argtype=`'static'`.
  Controls if uses the context maps to model social and physical interactions in the model.
  The default value is `1`.

### Vertical args

- `--K_train`, type=`int`, argtype=`'static'`.
  Number of multiple generations when training.
  The default value is `1`.
- `--K`, type=`int`, argtype=`'dynamic'`.
  Number of multiple generations when evaluating. The number of trajectories predicted for one agent is calculated by `N = args.K * args.Kc`, where `Kc` is the number of style channels.
  The default value is `1`.
- `--Kc`, type=`int`, argtype=`'static'`.
  Number of hidden categories used in alpha model.
  The default value is `20`.
- `--depth`, type=`int`, argtype=`'static'`.
  Depth of the random noise vector (for random generation).
  The default value is `16`.
- `--feature_dim`, type=`int`, argtype=`'static'`.
  (It is unused in this model) 
  The default value is `-1`.
- `--key_points`, type=`str`, argtype=`'static'`.
  A list of key-time-steps to be predicted in the agent model. For example, `'0_6_11'`.
  The default value is `'0_6_11'`.
- `--points`, type=`int`, argtype=`'static'`.
  Controls number of points (representative time steps) input to the beta model. It only works when training the beta model only.
  The default value is `1`.
<!-- DO NOT CHANGE THIS LINE -->

## Thanks

Codes of the Transformers used in this model comes from [TensorFlow.org](https://www.tensorflow.org/tutorials/text/transformer).  
Dataset files of ETH-UCY come from [SR-LSTM (CVPR2019)](https://github.com/zhangpur/SR-LSTM).  
Dataset split file of SDD comes from [SimAug (ECCV2020)](https://github.com/JunweiLiang/Multiverse).

## Contact us

Conghao Wong (@cocoon2wong): conghao_wong@icloud.com  
Beihao Xia (@NorthOcean): xbh_hust@hust.edu.cn
