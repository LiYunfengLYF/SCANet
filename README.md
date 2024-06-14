# SCANet

Our manuscript is available at [arxiv](https://arxiv.org/pdf/2406.07189)

## Project Paths Setup
You can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in `./data`. It should look like:
```
${PROJECT_ROOT}
  -- data
      -- LaSOT
          |-- ...
      -- TrackingNet
          |-- ...
      -- COCO
          |-- ...
      -- GOT10K
          |-- ...
      -- SARDet
          |-- ...
          ...
```
## Training
Download [SOT](https://github.com/botaoye/OSTrack) (OSTrack with SOT pretrained model) pretrained weights and put them under `$PROJECT_ROOT$/pretrained_models`.

```
python tracking/train.py --script scanet --config baseline --save_dir ./output --mode multiple --env_num 1 --nproc_per_node 2 --use_wandb 0
```

env_num doesn't need to be considered, it can be set to any number. if you want to train in different devices, you can consider it.

if you want to use env_num, go to lib/train/admin/local.py and lib/test/evaluation/local.py to set different device's num

## Evaluation

Checkpoint and raw results is coming soon

Download [checkpoint](https://drive.google.com/file/d/1XrSpF6plvnasbamIPTNtDc_qBN4xo9Ny/view?usp=sharing) and put it under `$PROJECT_ROOT$/output`.

```
python tracking/test_multi.py 
python tracking/eval.py
```
## Acknowledgments
Our project is developed upon [TBSI](https://github.com/RyanHTR/TBSI) and [OSTrack](https://github.com/botaoye/OSTrack). Thanks for their contributions which help us to quickly implement our ideas.

