This is official implementation (preview version) of [Dancing with Still Images: Video Distillation via Static-Dynamic Disentanglement](https://arxiv.org/abs/2312.00362). In this work, we provide the first systematic study of video distillation and introduce a taxonomy to categorize temporal compression. It first distills the videos into still images as static memory and then compensates the dynamic and motion information with a learnable dynamic memory block.
## Usage
Our method is a plug-and-play module.
1. Clone our repo.
```
git clone xxx
cd video_distillation
```
2. Prepare video datasets.
For convenience of use, we preprocess the video datasets into frames.
```
distill_utils
├── data
│   ├── HMDB51
│   │   ├── hmdb51_splits.csv
│   │   └── jpegs_112
│   ├── Kinetics
│   │   ├── broken_videos.txt
│   │   ├── replacement
│   │   ├── short_videos.txt
│   │   ├── test
│   │   ├── test.csv
│   │   ├── train
│   │   ├── train.csv
│   │   ├── val
│   │   └── validate.csv
│   ├── SSv2
│   │   ├── jpegs_112
│   │   ├── UCF101actions.pkl
│   └── UCF101
│       ├── jpegs_112
│       │       ├── v_ApplyEyeMakeup_g01_c01
│       │       ├── v_ApplyEyeMakeup_g01_c02
│       │       ├── v_ApplyEyeMakeup_g01_c03
│       │       └── ...
│       ├── UCF101actions.pkl
│       ├── ucf101_splits1.csv
│       └── ucf50_splits1.csv
└── ...

```
3. Static Learning
We use [DC]() for static learning. You can find DC code in this [repo]() and we provide code to load single frame data at utils.py and distill_utils/dataset.py. Or you can use [static memory]() trained by us.
4. Dynamic Fine-tuning
We have thoroughly documented the parameters employed in our experiments in [Suppl]().
For DM/DM+Ours
```
cd sh/baseline
# bash DM.sh GPU_num Dateset Learning_rate IPC
bash DM.sh 0 miniUCF101 100 1


# for DM+Ours
cd ../s2d
# bash MTT.sh GPU_num Dateset Learning_rate IPC
bash MTT.sh 0 miniUCF101 100 1
```

For MTT/MTT+Ours, it is necessary to first train the expert trajectory (refer [MTT]()).
```
cd sh/baseline
# bash buffer.sh GPU_num Dateset
bash buffer.sh 0 miniUCF101

# bash MTT.sh GPU_num Dateset Learning_rate IPC
bash MTT.sh 0 miniUCF101 100 1

cd ../s2d

```

...
# Todo