# **Eye gaze tracker (MLOps 2023 task)**
### *Artem Sokolov, 522 group*
## Train

By default, this command runs test training on a small part of data on CPU:
```
python3 eye_gaze_tracker/train.py
```
If you want to train on full data, you are to download the full dataset from
https://drive.google.com/drive/folders/1air9Yf578sx6tNwZdH42o1PcuDBEkZOo?usp=sharing

and put it into the *data/* folder. In this case, you should run:
```
python3 eye_gaze_tracker/train.py --config eye_gaze_tracker/config/full_train_cfg.json
```
The best model weights will be saved in *experiments/{model_name}/*

## Evaluation

By default, this command runs inference on images from *test/images* and saves results in *test/results*:
```
python3 eye_gaze_tracker/infer.py
```
Run next command for GPU inference:
```
python3 eye_gaze_tracker/infer.py --device cuda:0
```
To see other inference arguments run:
```
python3 eye_gaze_tracker/infer.py --h
```
