# ClothShape

#### 0. System
Upgrade system:
```
sudo apt-get update
sudo apt-get upgrade
```
#### 1. Nvidia Driver 
```
sudo apt install nvidia-415*
sudo reboot
```
#### 2. MiniConda
1. Download miniconda from the oficial [website](https://conda.io/miniconda.html). (Recommended: Python 3.* , 64-bits)
2. Install miniconda. (Recommended: use predefined paths and answer yes whenever you are asked yes/no)
    ```
    bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
    ```
#### 3. Dependencies
1. Create and activate conda environment for the project
    ```
    conda create -n ClothShape
    source activate ClothShape
    ```
2. Install Pytorch
    ```
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    ```
3. Install other rand dependencies
    ```
    conda install matplotlib opencv pillow scikit-learn scikit-image cython tqdm
    pip install tensorboardX
    ```
4. Deactivate environment
    ```
    conda deactivate
    ```
#### 4. Tensorboard
1. Create and activate conda environment for tensorboard: 
    ```
    conda create -n tensorboard python=3.6
    source activate tensorboard
    ```
2. Install Tensorflow CPU
    ```
    pip install tensorflow
    ```
3. Deactivate environment
    ```
    conda deactivate
    ```

## Run train
1. Run
    ```
    python src/train.py --exp_dir experiments/model1/basic_settings
    ```
3. To visualize. In a new terminal run:
    ```
    source activate tensorboard
    tensorboard --logdir path/to/repo/experiments/model1/basic_settings/
    ```

## Run test
1. Run
    ```
    python src/test.py --exp_dir experiments/model1/basic_settings
    ```
    results will be store in `experiments/model1/basic_settings/test`
