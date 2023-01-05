# CrowdRec

The code is tested on Windows10 and WSL2 (Windows Subsystem for Linux) with a NVIDIA GeForce RTX 3090 GPU and 64GB memory.
## Evaluation on GigaCrowd dataset

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

### Step 2. Pull code from docker hub
```
docker pull boycehbz/gigacrowd:latest
```

### Step 3. Run the code
```
docker container run --gpus all --rm \
-it -v <YOUR/PATH>/images:/workspace/data/GigaCrowd/images \
boycehbz/gigacrowd:latest
```
Then you can execute the reconstruction.
```
python main.py
```
You can also visualize the reconstructed meshes and rendered images by setting:
```
python main.py --save_mesh true --save_img true
```

When the fitting is finished, the ```predict.json``` can be generated in ```/workspace/output/results/results``` with the following command.
```
python generate_json.py
```
