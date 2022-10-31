# tinySSD-Pytorch

## 人工智能作业

tinySSD目标检测（个人平时作业）。

### 提交时间

2022年10月31日前。

### 作业描述&要求

**整理tinySSD代码。**

1. 按照数据、模型、训练、测试、可视化、自定义函数等拆分为多个文件。
2. 提供readme文件，说明清楚环境配置、训练流程
3. 提供简易测试代码及预训练模型
4. 贴出检测效果
5. 效果提升 …

**备注**：步骤1~4，占比80%；步骤5，占比20%。





## 代码

### 代码文件说明

```
tinySSD/
│
├── configs.py - 通用执行参数设置
├── train.py   - 训练模型
├── test.py    - 测试模型
│
├── requirements.txt - 环境配置要求
│
├── dataset/ - 数据集
│   ├── create_train.py - 用于数据集生成
│   ├── background/	- 背景图片
│   ├── sysu_train/	- 生成的训练数据
│   |	├── images
│   |	└── label.csv	- 标注
│   └── test/		- 测试图片
|
├── dataloader/ - 数据集加载
│    └─dataloader.py 
│
├── saved_figures/ - 保存训练指标变化图、测试图片的检测结果
│
├── saved_weights/ - 保存训练的模型权重
│
├── model/ - 模型文件
│   └─  tinySSD_model.py 
│   
└── utils/ 
    ├── loss.py - 计算损失
    └── visualizer.py - 可视化、作图函数
```



### 环境配置

#### 基本要求

- 计算机具备Nvidia显卡

- Python >= 3.6（推荐使用[Anaconda](https://www.anaconda.com/products/distribution)）

- Pytorch >= 1.12.1

- CUDA >= 11.6

  若已安装Anaconda，可直接在终端运行以下命令（可选执行，若当前环境已满足要求可跳过）：

  ```bash
  conda create -n tinySSD python=3.6
  conda activate tinySSD
  conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
  ```

#### 安装

1.将项目克隆到本地

```bash
git clone https://github.com/tsingfa/tinySSD.git
cd tinySSD
```

2.安装依赖环境

```bash
pip install -r requirements.txt
```

#### 数据准备

【注】：数据准备步骤已预先完成，可跳过此步骤，以下仅展示数据准备过程。

1.从[Releases]()下载数据集（dataset.rar）并解压至tinySSD工程文件夹下，数据集按如下形式分布：

```
├─dataset
  │  create_train.py
  │  
  ├─background
  │      000012-16347456122a06.jpg
  │	    ...
  │      191313-1519470793f344.jpg
  │      191328-15136820086f91.jpg
  │          
  ├─target
  │      0.png
  │      1.png
  │      
  └─test
         1.jpg
         2.jpg
```

2.运行create_train.py生成训练集

```bash
cd dataset
python create_train.py
```

运行完成后，结构如下：

```
├─dataset/
  │  create_train.py
  │  
  ├─background/
  │      000012-16347456122a06.jpg
  │	    ...
  │      191313-1519470793f344.jpg
  │      191328-15136820086f91.jpg
  │      
  ├─sysu_train/
  │  │	 label.csv
  │  │  
  │  └─images/
  │          000012-16347456122a06.jpg
  │		...
  │          183201-15420187218258.jpg
  │          
  ├─target/
  │      0.png
  │      1.png
  │      
  └─test/
         1.jpg
         2.jpg
```


### 训练流程

#### 训练



```bash

```



#### 测试





#### 测试结果