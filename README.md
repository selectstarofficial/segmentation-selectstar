# segmentation-selectstar

![](https://media.giphy.com/media/S7KnEAj0ZYpXeDLLuJ/giphy.gif)

Original repository: https://github.com/jfzhang95/pytorch-deeplab-xception

Modified to run NIA SurfaceMasking dataset by yoongi@selectstar.ai

# Training Surface Masking Dataset

1. Download NIA Surface Masking dataset from AIhub. (Not yet tested)
2. Generate mask images by running ```modules/utils/surface_dataset_tools/surface_polygon.py```, ```modules/utils/surface_dataset_tools/split_dataset.py```
3. Dataset structure should be like this.
    ```
    surface6
    ├── annotations
    │   ├── *.xml
    ├── images
    │   ├── *.jpg
    ├── masks
    │   ├── *.png
    ├── train.txt
    └── valid.txt
    ```
4. Install python packages
    ```
    Install Anaconda3 [https://www.anaconda.com/distribution/]
    conda create ml
    conda activate ml
    conda install conda
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
    pip install tensorboardx, matplotlib
    ```
5. Edit training options ```settings.py```
    ```
    Designate dataset directory
    ...
    elif dataset == 'surface':
        root_dir = '/home/super/Projects/dataset/surface6'
    ...
    ```
6. Run ```train.py``` 
    1. On Windows: ```python train.py```
    2. On Linux: ```python3 train.py```
    3. On multi-gpu: ```CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py```


# Predict
1. Prepare 'mp4 video' or 'jpg images' to predict. And put it into 'input' directory.
2. Prepare trained model like ```model_iou_77.pth.tar```
2. Edit ```RUN OPTIONS``` on predict.py
    ```
    MODEL_PATH, MODE, DATA_PATH, OUTPUT_PATH
    ```
3. Run ```predict.py```
4. Output result will be saved to OUTPUT_PATH


# Evaluate
1. Prepare dataset and trained model file.
2. Check settings.py options.
3. Run evaluate.py

### Performance
    ```
    [Validation]
    numImages:  2000
    Acc:0.9145832071304322, Acc_class:0.8474374571156529, mIoU:0.7729268004366087, fwIoU: 0.8433756131890291
    Loss: 17.542
    IoU of each class
    background: 0.8540012294525428
    bike_lane: 0.6477503591641897
    caution_zone: 0.5718921926153789
    crosswalk: 0.8020526751599193
    guide_block: 0.8133515872417134
    roadway: 0.856940048889669
    sidewalk: 0.8644995105328486
    ```
