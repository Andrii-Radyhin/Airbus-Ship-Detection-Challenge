# Airbus-Ship-Detection-Challenge

Abstract: this repo includes a pipeline using tf.keras for training UNet for the problem of ships detection.
Moreover, weights and for the trained model are provided.

**Important:** balanced dataset (dataset created during analysis) includes 4000 images per each class (0-15 ships) because original dataset contains ~80% images with no ships. Also dataset was downscaled to 256x256, with original resolution the metrics might be better.

## Plan of research

First, let's identify the main architecture:

 - Architecture: UNet
 - Loss function: DiceBCELoss, IoU
 - Optimizer: Adam (lr=1e-3, decay=1e-6)
 - learning scheduler: ReduceLROnPlateau(factor=0.5, patience=3)
 
 ## General thoughts
 
 Important to notice that we have dataset in run-length encoding format, utils/utils.py contains encoders/decoders which were based on 
 **Link to Kaggle Notebook:** [tap here](https://www.kaggle.com/paulorzp/run-length-encode-and-decode).
 
 I've tried DiceBCELoss and DiceLoss, IoU as loss.
 The best results have been obtained with DiceBCELoss in this case.
 
 I need to add I've been bounded with Cuda memory capacity, so basicaly I could not try batch size > 10 with original resolution. So, i said above, it was downscaled.
 
fullres_model.h5 contains trained original model + upscaling to original.

## Results
| Architecture | binary_accuracy | Input & Mask Resolution | Epochs |
| ------ | ------ | ------ | ------ | ------ |
| Unet | 0.958 | (256x256)  | 8 |

 Example 1:
 ![alt text](/images/pred1.png)
 
 Example 2: 
 ![alt text](/images/pred2.png)
 
 Example 3: 
 ![alt text](/images/pred3.png)
 
 Example 4: 
 ![alt text](/images/pred4.png)
 
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

