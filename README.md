# Airbus-Ship-Detection-Challenge

Abstract: this repo includes a pipeline using tf.keras for training UNet for the problem of ships detection.
Moreover, weights and for the trained model are provided. **I will use notebooks/train.ipynb as main file in EDA below, all images from output train.ipynb.**

**Important:** balanced dataset (dataset created during analysis) includes 4000 images per each class (0-15 ships) because original dataset contains ~80% images with no ships. Also dataset was downscaled to 256x256, with original resolution the metrics might be better.

## Guide & EDA
### Guide
First, let's deal with the dataset:

1. Important to notice that we have dataset in run-length encoding format, utils/utils.py contains encoders/decoders which were based on 
 **Link to Kaggle Notebook:** [tap here](https://www.kaggle.com/paulorzp/run-length-encode-and-decode).
 So, let's use it:
 
   a. We need to create a base dir (in my case it's named 'airbus-ship-detection'). Then put there two subdirs 'train_v2' and 'test_v2'.
 
   b. Next download dataset from kaggle: [tap here](https://www.kaggle.com/competitions/airbus-ship-detection/data). And unzip it in that two subfolders according to their names.
 
   c. Now in base dir must be something like this:
 <pre>
 ├── train_v2
 ├── test_v2
 ├── train_ship_segmentations_v2.csv
 ├── sample_submission_v2.csv
 </pre>
 
2. I will use jupyter notebook because it's easy way to use your graphics card via Anaconda Navigator and also virtual environment is safer.

   Here is useful video how to use your graphics card via Anaconda : [tap here](https://www.youtube.com/watch?v=r31jnE7pR-g).

   **It's important to say that train.ipynb or test.ipynb are independed notebooks (need only datasets).**
   
   I will explain next steps for non jupyter users, for jupyter users this step is only to add files from dir 'notebooks'.

   New and **final directory tree** should be like this:
<pre>
 ├── train_v2
 ├── test_v2
 ├── train_ship_segmentations_v2.csv
 ├── sample_submission_v2.csv
 ├── utils
 │   ├── losses.py
 │   └── utils.py
     └── generators.py
 ├── config.py
 ├── train.py         #or train.ipynb
 ├── test.py          #or test.ipynb
 ├── fullres_model & weights
     ├── fullres_model.h5
     └── seg_model_weights.best.hdf5
</pre>

     a. utils/losses.py file contains all losses (also custom metrics) that be used for this task.
       I prefer dice_p_bce for this task according to the results, but you can try any other. 
  
     b. utils/utils.py is encoders and decoders, data visualization and masks as image file.
       We will use it just in the next few steps.
  
     c. generators.py contains batch generators for model and predictions, ready to be shown via matplotlib, for test.py file.
     
     d. config.py is small but influential file for this project. It allows to control batch size, data scaling, epoch or number optimizer-steps per epoch. 
     
     e. train.py here we will train our model. test.py for results visualization.
     
     f. fullres_model.h5 768x768 input/output pretrained Unet model. As i said above, dataset was downscaled.
     So, seg_model_weights.best.hdf5 contains only weights for Unet 256x256.
     
3. **Necessary pips** (notebooks already contain them).

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

