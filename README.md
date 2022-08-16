# Airbus-Ship-Detection-Challenge

Abstract: this repo includes a pipeline using tf.keras for training UNet for the problem of ships detection.
Moreover, weights and for the trained model are provided. I will use notebooks/train.ipynb as main file in EDA below, all images from output train.ipynb.

**Important:** balanced dataset (dataset created during analysis) includes 4000 images per each class (0-15 ships) because original dataset contains ~80% images with no ships. Also dataset was downscaled to 256x256, with original resolution the metrics might be better.

## Plan of research & EDA
First, let's deal with the dataset:

1. Important to notice that we have dataset in run-length encoding format, utils/utils.py contains encoders/decoders which were based on 
 **Link to Kaggle Notebook:** [tap here](https://www.kaggle.com/paulorzp/run-length-encode-and-decode).
 So, let's use it:
 a. We need to create a base dir (in my case it's named 'airbus-ship-detection'). Then put there two subdirs 'train_v2' and 'test_v2'
 b. Next download dataset from kaggle: [tap here](https://www.kaggle.com/competitions/airbus-ship-detection/data). And unzip it in that two subfolders according to their names.
 c. Now in base dir must be something like this:
 <pre>
 ├── train_v2
 ├── test_v2
 </pre>
2. 




First, let's identify the main architecture:

 - Architecture: UNet
 - Loss function: DiceBCELoss, IoU
 - Optimizer: Adam (lr=1e-3, decay=1e-6)
 - learning scheduler: ReduceLROnPlateau(factor=0.5, patience=3)
 
 ## General thoughts
 
 
 
 I've tried DiceBCELoss and DiceLoss, IoU as loss.
 The best results have been obtained with DiceBCELoss in this case.
 
 I need to add I've been bounded with Cuda memory capacity, so basicaly I could not try batch size > 10 with original resolution. So, i said above, it was downscaled.
 
fullres_model.h5 contains trained original model + upscaling to original.

## Results
| Architecture | binary_accuracy | Input & Mask Resolution | Epochs |
| ------ | ------ | ------ | ------ |
| Unet | 0.958 | (256x256)  | 8 |

Example 1:
 ![alt text](images/pred1.PNG)
 
Example 2: 
 ![alt text](images/pred2.PNG)
 
Example 3: 
 ![alt text](images/pred3.PNG)
 
Example 4: 
 ![alt text](images/pred4.PNG)
 
 ## Installation

```sh
!pip install --user numpy
!pip install pandas
!python -m pip install -U matplotlib
!python -m pip install -U scikit-image
!pip install -U scikit-learn
!pip install keras
!pip install tensorflow
```

Or you can also use requerements.txt.

