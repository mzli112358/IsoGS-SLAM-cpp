<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">SplaTAM: 使用3D高斯点进行密集RGB-D SLAM的Splat、跟踪与建图</h1>
  <h3 align="center">CVPR 2024</h3>
  <p align="center">
    <a href="https://nik-v9.github.io/"><strong>Nikhil Keetha</strong></a>
    ·
    <a href="https://jaykarhade.github.io/"><strong>Jay Karhade</strong></a>
    ·
    <a href="https://krrish94.github.io/"><strong>Krishna Murthy Jatavallabhula</strong></a>
    ·
    <a href="https://gengshan-y.github.io/"><strong>Gengshan Yang</strong></a>
    ·
    <a href="https://theairlab.org/team/sebastian/"><strong>Sebastian Scherer</strong></a>
    <br>
    <a href="https://www.cs.cmu.edu/~deva/"><strong>Deva Ramanan</strong></a>
    ·
    <a href="https://www.vision.rwth-aachen.de/person/216/"><strong>Jonathon Luiten</strong></a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/pdf/2312.02126.pdf">论文</a> | <a href="https://youtu.be/jWLI-OFp3qU">视频</a> | <a href="https://spla-tam.github.io/">项目页面</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./assets/1.gif" alt="Logo" width="100%">
  </a>
</p>

<br>

## 敬请期待更快更好的SplaTAM变体！

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>目录</summary>
  <ol>
    <li>
      <a href="#installation">安装</a>
    </li>
    <li>
      <a href="#demo">在线演示</a>
    </li>
    <li>
      <a href="#usage">使用方法</a>
    </li>
    <li>
      <a href="#downloads">数据下载</a>
    </li>
    <li>
      <a href="#benchmarking">基准测试</a>
    </li>
    <li>
      <a href="#acknowledgement">致谢</a>
    </li>
    <li>
      <a href="#citation">引用</a>
    </li>
    <li>
      <a href="#developers">开发者</a>
    </li>
  </ol>
</details>

## 安装

##### （推荐）
SplaTAM已在Python 3.10、Torch 1.12.1和CUDA=11.6环境下进行基准测试。但是，Torch 1.12不是硬性要求，代码也已在其他版本的Torch和CUDA上测试过，例如Torch 2.3.0和CUDA 12.1。

安装所有依赖的最简单方法是使用[anaconda](https://www.anaconda.com/)和[pip](https://pypi.org/project/pip/)，按以下步骤操作：

```bash
conda create -n splatam python=3.10
conda activate splatam
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

<!-- Alternatively, we also provide a conda environment.yml file :
```bash
conda env create -f environment.yml
conda activate splatam
``` -->

#### Windows

在Windows上使用Git bash进行安装，请参考[Issue#9中分享的说明](https://github.com/spla-tam/SplaTAM/issues/9#issuecomment-1848348403)。

#### Docker和Singularity设置

我们还提供了docker镜像。我们建议在docker镜像内使用venv来运行代码：


```bash
docker pull nkeetha/splatam:v1
bash bash_scripts/start_docker.bash
cd /SplaTAM/
pip install virtualenv --user
mkdir venv
cd venv
virtualenv --system-site-packages splatam
source ./splatam/bin/activate
pip install -r venv_requirements.txt
```

设置singularity容器类似：
```bash
cd </path/to/singularity/folder/>
singularity pull splatam.sif docker://nkeetha/splatam:v1
singularity instance start --nv splatam.sif splatam
singularity run --nv instance://splatam
cd <path/to/SplaTAM/>
pip install virtualenv --user
mkdir venv
cd venv
virtualenv --system-site-packages splatam
source ./splatam/bin/activate
pip install -r venv_requirements.txt
```

## 演示

### 在线

您可以使用iPhone或配备LiDAR的Apple设备，通过下载并使用<a href="https://apps.apple.com/au/app/nerfcapture/id6446518379">NeRFCapture</a>应用程序来使用SplaTAM重建您自己的环境。

确保您的iPhone和PC连接到同一个WiFi网络，然后运行以下命令：

 ```bash
bash bash_scripts/online_demo.bash configs/iphone/online_demo.py
```

在应用程序中，持续点击发送以获取连续帧。一旦帧捕获完成，应用程序将与PC断开连接，您可以在PC上查看SplaTAM的交互式重建渲染！以下是一些很棒的示例结果：

<p align="center">
  <a href="">
    <img src="./assets/collage.gif" alt="Logo" width="75%">
  </a>
</p>

### 离线

您也可以先捕获数据集，然后使用以下命令在数据集上离线运行SplaTAM：

```bash
bash bash_scripts/nerfcapture.bash configs/iphone/nerfcapture.py
```

### 数据集收集

如果您只想使用NeRFCapture应用程序捕获自己的iPhone数据集，请使用以下命令：

```bash
bash bash_scripts/nerfcapture2dataset.bash configs/iphone/dataset.py
```

## 使用方法

我们将使用iPhone数据集作为示例来展示如何使用SplaTAM。以下步骤与其他数据集类似。

要运行SplaTAM，请使用以下命令：

```bash
python scripts/splatam.py configs/iphone/splatam.py
```

要可视化最终的交互式SplaTAM重建，请使用以下命令：

```bash
python viz_scripts/final_recon.py configs/iphone/splatam.py
```

要以在线方式可视化SplaTAM重建，请使用以下命令：

```bash
python viz_scripts/online_recon.py configs/iphone/splatam.py
```

要将splats导出为.ply文件，请使用以下命令：

```bash
python scripts/export_ply.py configs/iphone/splatam.py
```

`PLY`格式的Splats可以在[SuperSplat](https://playcanvas.com/supersplat/editor)和[PolyCam](https://poly.cam/tools/gaussian-splatting)等查看器中可视化。

要在SplaTAM重建上运行3D高斯点渲染，请使用以下命令：

```bash
python scripts/post_splatam_opt.py configs/iphone/post_splatam_opt.py
```

要在使用真实位姿的数据集上运行3D高斯点渲染，请使用以下命令：

```bash
python scripts/gaussian_splatting.py configs/iphone/gaussian_splatting.py
```

## 数据下载

DATAROOT默认为`./data`。如果数据集存储在您机器上的其他位置，请在场景特定的配置文件中更改`input_folder`路径。

### Replica

按以下方式下载数据，数据将保存到`./data/Replica`文件夹中。请注意，Replica数据由iMAP的作者生成（但由NICE-SLAM的作者托管）。如果您使用该数据，请引用iMAP。

```bash
bash bash_scripts/download_replica.sh
```

### TUM-RGBD

```bash
bash bash_scripts/download_tum.sh
```

### ScanNet

请按照[ScanNet](http://www.scan-net.org/)网站上的数据下载程序，并使用此[代码](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py)从`.sens`文件中提取颜色/深度帧。

<details>
  <summary>[ScanNet目录结构（点击展开）]</summary>

```
  DATAROOT
  └── scannet
        └── scene0000_00
            └── frames
                ├── color
                │   ├── 0.jpg
                │   ├── 1.jpg
                │   ├── ...
                │   └── ...
                ├── depth
                │   ├── 0.png
                │   ├── 1.png
                │   ├── ...
                │   └── ...
                ├── intrinsic
                └── pose
                    ├── 0.txt
                    ├── 1.txt
                    ├── ...
                    └── ...
```
</details>


我们使用以下序列：
```
scene0000_00
scene0059_00
scene0106_00
scene0181_00
scene0207_00
```

### ScanNet++

请按照<a href="https://kaldir.vc.in.tum.de/scannetpp/">ScanNet++</a>网站上的数据下载和图像去畸变程序进行操作。
此外，对于去畸变DSLR深度图像，我们使用<a href="https://github.com/Nik-V9/scannetpp">官方ScanNet++处理代码的我们自己的变体</a>。我们很快将向官方ScanNet++存储库提交pull request。

我们使用以下序列：

```
8b5caf3398
b20a261fdf
```

对于b20a261fdf，我们使用前360帧，因为在第360帧之后轨迹出现突然跳跃/传送。请注意，ScanNet++主要是作为NeRF训练和新视角合成数据集。

### Replica-V2

我们使用来自vMAP的Replica-V2数据集来评估新视角合成。请从<a href="https://github.com/kxhit/vMAP">vMAP</a>下载预生成的replica序列。

## 基准测试

对于运行SplaTAM，我们建议使用[weights and biases](https://wandb.ai/)进行日志记录。这可以通过在配置文件中将`wandb`标志设置为True来开启。同时确保指定路径`wandb_folder`。如果您没有wandb账户，请先创建一个。请确保将`entity`配置更改为您的wandb账户。每个场景都有一个配置文件夹，需要指定`input_folder`和`output`路径。

下面，我们展示每个数据集中一个场景的示例运行命令。SLAM之后，将评估轨迹误差以及渲染指标。结果默认保存到`./experiments`。

### Replica

要在`room0`场景上运行SplaTAM，请运行以下命令：

```bash
python scripts/splatam.py configs/replica/splatam.py
```

要在`room0`场景上运行SplaTAM-S，请运行以下命令：

```bash
python scripts/splatam.py configs/replica/splatam_s.py
```

对于其他场景，请修改`configs/replica/splatam.py`文件或使用`configs/replica/replica.bash`。

### TUM-RGBD

要在`freiburg1_desk`场景上运行SplaTAM，请运行以下命令：

```bash
python scripts/splatam.py configs/tum/splatam.py
```

对于其他场景，请修改`configs/tum/splatam.py`文件或使用`configs/tum/tum.bash`。

### ScanNet

要在`scene0000_00`场景上运行SplaTAM，请运行以下命令：

```bash
python scripts/splatam.py configs/scannet/splatam.py
```

对于其他场景，请修改`configs/scannet/splatam.py`文件或使用`configs/scannet/scannet.bash`。

### ScanNet++

要在`8b5caf3398`场景上运行SplaTAM，请运行以下命令：

```bash
python scripts/splatam.py configs/scannetpp/splatam.py
```

要在`8b5caf3398`场景上运行新视角合成，请运行以下命令：

```bash
python scripts/eval_novel_view.py configs/scannetpp/eval_novel_view.py
```

对于其他场景，请修改`configs/scannetpp/splatam.py`文件或使用`configs/scannetpp/scannetpp.bash`。

### ReplicaV2

要在`room0`场景上运行SplaTAM，请运行以下命令：

```bash
python scripts/splatam.py configs/replica_v2/splatam.py
```

要在SplaTAM之后的`room0`场景上运行新视角合成，请运行以下命令：

```bash
python scripts/eval_novel_view.py configs/replica_v2/eval_novel_view.py
```

对于其他场景，请修改配置文件。

## 致谢

我们感谢以下存储库的作者提供开源代码：

- 3D高斯点
  - [Dynamic 3D Gaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians)
  - [3D Gaussian Splating](https://github.com/graphdeco-inria/gaussian-splatting)
- 数据加载器
  - [GradSLAM & ConceptFusion](https://github.com/gradslam/gradslam/tree/conceptfusion)
- 基线
  - [Nice-SLAM](https://github.com/cvg/nice-slam)
  - [Point-SLAM](https://github.com/eriksandstroem/Point-SLAM)

## 引用

如果您发现我们的论文和代码有用，请引用我们：

```bib
@inproceedings{keetha2024splatam,
        title={SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM},
        author={Keetha, Nikhil and Karhade, Jay and Jatavallabhula, Krishna Murthy and Yang, Gengshan and Scherer, Sebastian and Ramanan, Deva and Luiten, Jonathon},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2024}
      }
```

## 开发者
- [Nik-V9](https://github.com/Nik-V9) ([Nikhil Keetha](https://nik-v9.github.io/))
- [JayKarhade](https://github.com/JayKarhade) ([Jay Karhade](https://jaykarhade.github.io/))
- [JonathonLuiten](https://github.com/JonathonLuiten) ([Jonathan Luiten](https://www.vision.rwth-aachen.de/person/216/))
- [krrish94](https://github.com/krrish94) ([Krishna Murthy Jatavallabhula](https://krrish94.github.io/))
- [gengshan-y](https://github.com/gengshan-y) ([Gengshan Yang](https://gengshan-y.github.io/))

