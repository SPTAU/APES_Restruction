# APES: Attention-based Point Cloud Edge Sampling

<p>
<a href="https://arxiv.org/pdf/2302.14673.pdf">
    <img src="https://img.shields.io/badge/PDF-arXiv-brightgreen" /></a>
<a href="https://junweizheng93.github.io/publications/APES/APES.html">
    <img src="https://img.shields.io/badge/Project-Homepage-red" /></a>
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Framework-PyTorch-orange" /></a>
<a href="https://mmengine.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/Framework-MMEngine-ff69b4" /></a>
<a href="https://github.com/JunweiZheng93/APES/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
</p>


## Homepage

This project is selected as a Highlight at CVPR 2023! For more information about the project, please refer to our [project homepage](https://junweizheng93.github.io/publications/APES/APES.html).


## Prerequisites

Install all necessary packages using:

```shell
conda install pytorch==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```


## Data

Download and preprocess the data using:

```shell
python utils.download_modelnet.py  # for classification
```


## Train

Train models from scratch using:

```shell
# using single GPU
# command: bash utils/single_gpu_train.sh cfg_file
bash utils/single_gpu_train.sh configs/apes/apes_cls_local-modelnet-200epochs.py  # for classification using local-based downsampling
bash utils/single_gpu_train.sh configs/apes/apes_cls_global-modelnet-200epochs.py  # for classification using global-based downsampling

# using multiple GPUs 
# command: bash utils/dist_train.sh cfg_file num_gpus
bash utils/dist_train.sh configs/apes/apes_cls_local-modelnet-200epochs.py 4  # for classification using local-based downsampling
bash utils/dist_train.sh configs/apes/apes_cls_global-modelnet-200epochs.py 4  # for classification using global-based downsampling
```


## Test

Test model with checkpoint using:

```shell
# using single GPU
# command: bash utils/single_gpu_test.sh cfg_file ckpt_path
bash utils/single_gpu_test.sh configs/apes/apes_cls_local-modelnet-200epochs.py ckpt_path  # for classification using local-based downsampling
bash utils/single_gpu_test.sh configs/apes/apes_cls_global-modelnet-200epochs.py ckpt_path # for classification using global-based downsampling

# using multiple GPUs 
# command: bash utils/dist_test.sh cfg_file ckpt_path num_gpus
bash utils/dist_test.sh configs/apes/apes_cls_local-modelnet-200epochs.py ckpt_path 4  # for classification using local-based downsampling
bash utils/dist_test.sh configs/apes/apes_cls_global-modelnet-200epochs.py ckpt_path 4  # for classification using global-based downsampling
```


## Citation

If you are interested in this work, please cite as below:

```text
@article{wu2023apes,
  title={Attention-based Point Cloud Edge Sampling},
  author={Wu, Chengzhi and Zheng, Junwei and Pfrommer, Julius and Beyerer, Juergen},
  journal={arXiv preprint arXiv:2302.14673},
  year={2023}
}
```