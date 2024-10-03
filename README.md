# Combating Data Imbalances in Federated Semi-supervised Learning with Dual Regulators (FedDure)
![FedDure](framework.png)
## Setup
```
conda create --name fssl python=3.6.8
conda activate clip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install git+https://github.com/openai/CLIP.git](https://github.com/EasyFL-AI/EasyFL.git
```
Alternatively, you can use `pip install -r requirements.txt` to install all the dependencies.

## Download Dataset
We experiment with three datasets: Cifar10, CINIC10 and FashionMNIST.

## Training
We provide the running scripts in `configs`, which allow you to reproduce the results on our paper.
Make sure you change the data path (`root`) in `.yaml` files.

### Cifar10
**Slurm cluster**:

- sh train_slurm.sh vi_irdc 1 ./configs/cifar10/res9_meta_pseudo_cifar10_wnet1_dir_dir_5.yaml

**Workstation**

- bash train.sh ./configs/cifar10/res9_cifar10_dir_dir_5.yaml

You run the commands for CINIC10 in a similar manner by changing the scripts.
## References
If you use this code, please cite
```
@article{bai2023combating,
  title={Combating Data Imbalances in Federated Semi-supervised Learning with Dual Regulators},
  author={Sikai Bai, Shuaicheng Li,Weiming Zhuang, Kunlin Yang, Jie Zhang, Jun Hou, Shuai Yi, Shuai Zhang, Junyu Gao},
  conference={The Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24)},
  year={2023}
}
```
