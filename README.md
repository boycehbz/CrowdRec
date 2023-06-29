# CrowdRec

The code for the 1st place solution of [2022 GigaCrowd challenge](https://gigavision.cn/track/track/?nav=GigaCrowd&type=nav&t=1678150678599).<br>

> CrowdRec: 3D Crowd Reconstruction from Single Color Images<br>
> [Buzhen Huang](http://www.buzhenhuang.com/), Jingyi Ju, [Yangang Wang](https://www.yangangwang.com/#me)<br>
> \[[Paper](https://arxiv.org/pdf/2110.10355.pdf)\]<br>

The code is tested on Windows 10 and WSL2 (Windows Subsystem for Linux) with an NVIDIA GeForce RTX 3090 GPU and 64GB memory.<br>

## Demo

### Installation

### Getting Started
```
python demo.py --config cfg_files/demo.yaml
```


## Evaluation on GigaCrowd dataset with Docker

### Step 1. Download dataset
Download the test set from [GigaVision challenge website](https://www.gigavision.cn/track/track?nav=GigaCrowd&type=nav) and put the images in <YOUR/PATH> with the following structure.

```
----images
    |---playground1_00
        |---playground1_00_000000.jpg
        |---playground1_00_001740.jpg
        |---...
    |---playground1_01
    |---stadiumEntrance_00
    |---stadiumEntrance_01
```

### Step 2. Load docker image
You can obtain the docker image from [here](https://pan.baidu.com/s/1Ye9VQ3vOx4qahSOdY81q4w?pwd=k77l).
```
tar -zxvf gigacrowd.tar.gz
```
```
docker load < gigacrowd.tar
```

### Step 3. Run the code
```
docker container run --gpus all --rm \
-it -v <YOUR/PATH>/images:/workspace/data/GigaCrowd/images \
boycehbz/gigacrowd:latest
```
Then you can execute the reconstruction. The SMPL parameters for each person will be saved in ```/workspace/output/final/gigacrowd/images```.
```
python main.py
```

When the fitting is finished, the ```predict.json``` can be generated in ```/workspace/output/results/results``` with the following command.
```
python generate_json.py
```
