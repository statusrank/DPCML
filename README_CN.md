# [The Minority Matters: A Diversity-Promoting Collaborative Metric Learning Algorithm](https://scholar.google.com.hk/citations?view_op=view_citation&hl=zh-CN&user=5ZCgkQkAAAAJ&citation_for_view=5ZCgkQkAAAAJ:UeHWp8X0CEIC)

## 环境依赖

所有实验都在一台配备了ubuntu 18.04操作系统、Intel(R) Xeon(R) Gold 6246R CPU@3.40GHz处理器和RTX 3090显卡的服务器上进行，相关依赖如下（见code/requirements.txt）：
- python 3.8+
- pytorch 1.8+
- numpy
- tqdm
- toolz 
- scikit-learn
- scipy
- pands


## 数据集

论文中主要涉及到以下四个数据集，所有数据均存放至data/文件夹下，具体如下：
- MovieLens-1M
- CiteULike
- Steam-200k
- MovieLens-10M

## 代码运行

### 安装环境依赖

执行如下命令，安装依赖：
```
pip3 install -r requirements.txt
```

### 运行

可采用如下方式 (COCML对应论文中的DPCML1 （此时sampling_strategy必须设置为uniform）， HarCML对应论文中的DPCML2（此时（此时sampling_strategy必须设置为hard）)：
```
  CUDA_VISIBLE_DEVICES=0 python3 train_best.py \
    --data_path=data/dataset_name \
    --model=model_name \
    --margin=1.0 \
    --lr=1e-3 \
    --per_user_k=5 \
    --sampling_strategy=uniform \
    --dim=100 \
    --reg=10  \
    --epoch=100 \
    --m1=0.1 \
    --m2=0.35
```
也可通过配置shell脚本运行：
```shell
chmod +x run.sh
./run.sh
```

## 实验结果
实验过程中会进行日志保存，包括每个epoch的训练、验证、测试的相应指标。 可以看出实验结果与论文中基本一致。