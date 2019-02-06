# Unet in Image Boundary Detection
This repository contains source code to implement a modified Unet framework on image boundary detection, an image segmentation task.

## Demonstration
We illustrate the functionality of this project as follows.    

Image | Mask | Prediction
:-------------------------:|:-------------------------:|:-------------------------:
![](/demo/image.PNG) | ![](/demo/mask.PNG) | ![](/demo/label.PNG)
Original images where we try to detect boundaries(red sections). | Manually labelled images (yellow lines). | The predicted boundaries by the model (white lines).

## Model Framework    
We mainly employed the framework of [U-Net](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015), a Convolutional Neural Network as follows.
![](/demo/unet.PNG)

## Datasets
Data are saved in `.p` files (python pickle files) as follows.    
`images_rec.p`: contains raw microscope photos provided by Leyao Shen;    
`masks_rec.p`: containslabeled images provided by Leyao Shen;    
`labels_rec.p`: contains only labels generated by `preproc.py`.

## Scripts
`model.py`: defines the Unet framework;    
`obj.py`: defines loss functions and evaluation metrics;    
`preproc.py`: performs data augmentation and label generation;    
`main.py`: trains and evaluates model and makes predictions.
