# [The Minority Matters: A Diversity-Promoting Collaborative Metric Learning Algorithm](https://scholar.google.com.hk/citations?view_op=view_citation&hl=zh-CN&user=5ZCgkQkAAAAJ&citation_for_view=5ZCgkQkAAAAJ:UeHWp8X0CEIC)

## 环境依赖

**基本要求：** 此部分应写明（可复现论文结果的）代码运行所需的环境及依赖，以确保结果的可复现性，包括但不限于:
- 服务器的CPU、GPU型号
- 操作系统及版本
- python环境 （如果有）
- pytorch、tensorflow等版本（如果有）
- 其他第三方库，如numpy、pandas、scikit-learn的版本（如果有）


## 数据集

**基本要求：** 
1. 此处需列出论文中所涉及到的所有数据集（所涉及到的数据集都需公开并上传）。
2. 若数据集为公开数据集，可附上（有效的）公开数据集下载链接，并在此说明各个数据集的具体处理细节，以及将下载数据存放到哪个文件夹下以保证代码的正常运行。
3. 若数据集为在公开数据集基础上进一步构造所得，需上传该数据集或附上（有效的）公开数据集下载链接，并在此说明各个数据集的具体处理细节或给出数据集处理代码及运行方式。

## 代码运行

### 安装环境依赖

**基本要求：** 此处应写明如何安装每个环境依赖，以确保实验的正常运行。例如，将所有依赖存入requirements.txt文件夹中并执行（其他方式皆可）：
```
pip3 install -r requirements.txt
```

### 运行
**基本要求：** 此处不要求特定的运行方式，但应给出每个数据集的（能复现出论文实验结果的）代码运行命令，需写明每个数据集上的准确超参数，也可以不同数据集采用不同的运行方式：
1. 若采用argparse进行超参数，可采用如下方式：
```
  CUDA_VISIBLE_DEVICES=3 python3 train.py dataset-name method-name --lr=* --reg=*
```
2. 若采用json/yaml等方式配置超参数，需说明每个数据集读入哪个配置文件
3. 也可通过shell脚本一键运行

例子：
#### CIFAR-10
```shell
CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-10-long-tail --lr=0.1 ...
```
#### CIFAR-100
```shell
CUDA_VISIBLE_DEVICES=0 python3 train.py cifar-100-long-tail --lr=0.3 ...
```

#### shell脚本
```shell
chmod +x run.sh
./run.sh
```

## 实验结果
**基本要求：** 此处应简单说明实验运行的输出、实验的最终结果以及与论文中实验结果的一致性。

```
具体例子可见code-example
```