# Integrating Contrastive Learning with Dynamic Modelsfor Reinforcement Learning from Images   

This repository is the official pytorch implementation of CoDy for the DeepMind control experiments. Our implementation of SAC is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae).   

## Requirements  
It is simple to install required dependencies by running:  
```sh
conda env create -f environment.yml  
```
Then you can activate the environment by running:  
```sh
source activate py3.6  
```
## Instructions
To train a CoDy+SAC agent on the ```sh cartpole swingup ``` task from images, you can run:
```sh
python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --action_repeat 8 \
    --save_tb \
    --frame_stack 3 \
    --seed -1 \
    --eval_freq 10000 \
    --batch_size 256 \
    --num_train_steps 1000000 
```
