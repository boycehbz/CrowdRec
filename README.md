# CrowdRec

## Evaluation on GigaCrowd dataset

### Step 1. Download dataset
Download the test set from [GigaVision challenge website](https://www.gigavision.cn/track/track?nav=GigaCrowd&type=nav) and put the images in <YOUR/PATH> with following structure.

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
docker container run --gpus all --rm -it -v <YOUR/PATH>/images:/workspace/data/GigaCrowd/images boycehbz/gigacrowd:latest
```

```
python main.py
```
You can also visualize the reconstructed meshes and overlay images by setting:
```
python main.py --save_mesh true --save_img true
```

When the fitting is finished, you can generate ```predict.json``` from the fitted results with the following command.
```
python generate_json.py
```
