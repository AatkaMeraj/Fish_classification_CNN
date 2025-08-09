# Fish_classification_CNN

**PROJECT DESCRIPTION**

**The goal of this project is to classify fish species accurately using images. It involves preprocessing images, training convolutional neural networks (CNNs), and evaluating their performance. The classification can be used for automated species recognition, aiding scientific research and conservation efforts.**

**INSTALLATION**

Clone the Repo:

https://github.com/AatkaMeraj/Fish_classification_CNN.git

**MODEL ARCHITECTURE**

Models Trained:

Base model: [ResNet50, EfficientNetB0, VGG16, MobileNet, InceptionV3] pretrained on ImageNet

Custom dense layers for classification:

Input image size:  224x224 pixels and 299x299 pixles for InceptionV3

Loss function: Categorical Cross-Entropy

Optimizer: Adam

**TRAINING**

Number of epochs: 20  and fine-tuning epochs: 10(adjustable)

Batch size: 32

Data augmentation applied for better generalization: rotation, flipping, zoom, etc.

Early stopping, ReduceLROnPLateau and model checkpoint callbacks used.

**RESULT**

VGG16 Model: 97% accuracy

EfficientNetB0: 96% accuracy

MobileNet: 100% accuracy

ResNet50: 100% accuracy

InceptionV3: 98% accuracy

**DEPLOYMENT**

**InceptionV3 model is used for deployment.**

InceptionV3 model is used as it is not over-fitting and the accuracy is also good among other models.




