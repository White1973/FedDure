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
@inproceedings{bai2024combating,
  title={Combating data imbalances in federated semi-supervised learning with dual regulators},
  author={Bai, Sikai and Li, Shuaicheng and Zhuang, Weiming and Zhang, Jie and Yang, Kunlin and Hou, Jun and Yi, Shuai and Zhang, Shuai and Gao, Junyu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={10},
  pages={10989--10997},
  year={2024}
}
```
