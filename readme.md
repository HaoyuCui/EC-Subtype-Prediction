Endometrial Cancer Subtype Prediction
===========

Official code release

[Journal link](tbd) | [Cite](tbd)


## Installation

Install the dependencies 

```bash
pip install -r requirements.txt
```

## Preprocessing

1. We have uploaded another repo for data preprocessing: [WSI_Segmenter](https://github.com/HaoyuCui/WSI_Segmenter). Which can also be found in the [./preprocess](./preprocess) directory. The detailed patch extraction and segmentation steps can be found in the [./preprocess/readme.md](preprocess/readme.md). 

2. Extract raw patches to 1024x1024 resolution, use [tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox) or [DeepZoom](https://github.com/ncoudray/DeepPATH/blob/master/DeepPATH_code/00_preprocessing/0b_tileLoop_deepzoom6.py) for patch extraction. The tumor segmentation network can be easily added to these pipelines.


## Data preparation

1. Prepare the data in the following structure, png or jpeg format is supported. Note that extracting patches only from the tumor region is recommended.

    ```markdown
    ├── data
    │   ├── slide_1
    │   │   ├── patch_1.png
    │   │   ├── patch_2.png
    │   │   ├── ...
    │   ├── slide_2
    │   │   ├── patch_1.png
    │   │   ├── patch_2.png
    │   │   ├── ...
    │   ├── ...
    │   └── slide_n
    │       ├── ...
    │       └── patch_n.png
    ```

2. **[Optional]** Run stain normalization for whole slide image tiles [wsi_normalizer](https://github.com/HaoyuCui/WSI_Normalizer). This step is recommended for external validation when you have multiple cohorts.

3. Organize your data like `example.csv`. Create k-fold split for the data.

    ```bash
    python utils/gen_kfold_split.py --csv <CSV_PATH>  --dir <STEP_2_OUTPUT_DIR> --k 5 --on slide
    ```
    
    `--on slide` split the data on slide level
    
    `--on patient` split the data on patient level (use name column)
   
   A directory named `kf` will be created in the current directory.

5. Modify the [config.yaml](config.yaml) file to set hyperparameters and UNI's storage path.

    - Hyperparameters: **batch_size**, **lr**, **epochs**, **iters_to_val**, **save_best**
    
    - Task-specific config: **class_names**

## Train and evaluate

1. Train & evaluate a single fold (e.g., fold 1) and evaluate on the validation set
    ```bash
    python train.py --fold 1
    ```

2. Train & evaluate all folds (for Windows)
    ```bash
    python ./scripts/train_kf.py
    ```
   Train & evaluate all folds (for Linux)
    ```bash
    sh ./scripts/train_kf.sh
    ```

3. The results will be saved in the `runs/` directory.

   In the format of:
   ```txt
    ├── runs
    │   ├── {cmbs}_{freeze_ration}  # configuration
    │   │   ├── 1  # fold name
    │   │   │   ├── {fold}_best.pth  # best model
    │   │   │   ├── slide_{iter}.png  # slide-level ROC
    │   │   │   ├── ...
    │   │   ├── ...
    │   ...
   ```
   
4. [im4MEC pipeline](https://github.com/AIRMEC/im4MEC) Here, we follow the im4MEC pipeline to pretrain, train and evaluate the model (we simplify the original code for single-gpu-support). The codes are provided in the `im4MEC-pipeline/` directory.

    - Pretrain the model:
    ```bash
    python train_moco_v2.py
    ```

    - Extract features:
    ```bash
    python extract_features.py
    ```
   
   Then, you can use the CLAM pipeline for model training and evaluation.
   

## Comparison experiments

We are grateful to the authors for sharing their code. We use CLAM for data preprocessing and feature extraction in comparison experiments.

| Model      | Authors          | GitHub link                                             |
|---------------|---------------|---------------------------------------------------------|
| CLAM          | Lu et al.     | [https://github.com/mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM) |
| im4MEC        | Fremond et al.| [https://github.com/AIRMEC/im4MEC](https://github.com/AIRMEC/im4MEC) |


## License

© [IMIC](https://imic.nuist.edu.cn/) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

## Reference

TBD

