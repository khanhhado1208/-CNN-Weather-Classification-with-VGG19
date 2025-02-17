# CNN Weather Classification with VGG19
## Multi-class Weather Classification using Transfer Learning (VGG19). Implemented a CNN model using VGG19 to classify weather conditions (Cloudy, Rain, Shine, Sunrise).

### 📌 Introduction
In this exercise, we use the VGG19 model with transfer learning to identify 4 types of weather based on images: 
☁️ Cloudy
🌧 Rain
☀ Shine
🌄 Sunrise

### Objective: 
✅ Train a CNN model with high accuracy
✅ Avoid overfitting and underfitting
✅ Compare performance when freezing and fine-tuning VGG19

### Dataset
Source: Multi-class Weather Dataset [https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset]
Number of photos: 1125 photos in 4 classes
Image size: 224x224 pixels
Preprocessing:
Normalize data: Divide pixel value by 255
One-hot encoding for labels
Split the data set: 70% training, 30% validation

### Results & Reviews
| Method                              | Train Accuracy | Validation Accuracy | Test Accuracy | Overfitting                                                         |
|-------------------------------------|----------------|---------------------|---------------|---------------------------------------------------------------------|
| Train all layers (trainable = True) | 100%           | 94.97%              | 94.97%        | There are signs of overfitting (Train is 5% higher than Validation) |
| Freeze VGG19 (trainable = False)    | 95.7%          | 87.57%              | 87.57%        | Less overfitting, but less learning ability                         |

#### Trainable model = True (100% train - 94.97% validation)
+ When setting layer.trainable = True, the entire VGG19 is retrained from scratch, including the weights of the convolutional layers.
+ It helps the model learn better on the training set, helping to increase the highest accuracy, but there is also a higher risk of overfitting if the data is not diverse enough.
+ The model learns very well on the training set (reaches 100% accuracy). High validation accuracy (94.97%), good generalization model.
+ Problem: Slight overfitting, train is too high (100%) while validation is only 94.97%. It takes a long time to train, because all the weights of the model have to be updated, requiring more GPU resources.

#### Trainable = False (95.7% train - 87.57% validation)
+ When freezing the entire VGG19, only the fully connected layers behind are trained.
+ It helps avoid overfitting, but limits learning because the model only uses existing features from ImageNet, without further refinement.
+ Train is faster, because it only trains fully connected layers. Validation accuracy is stable (87.57%), although lower than trainable = True.
+ Problem: Mild underfitting, because the model doesn't learn the best for the data. Accuracy on test is lower (~87.57%), because the model only uses pre-trained features.

### Confusion Matrix

<img width="370" alt="image" src="https://github.com/user-attachments/assets/e45e2f9f-0f3c-4c67-8de3-1c075a3f0760" />

| True Label / Predicted label | Cloudy | Rain | Shine | Sunrise |
|------------------------------|--------|------|-------|---------|
| Cloudy                       | 90     | 1    | 0     | 0       |
| Rain                         | 0      | 72   | 0     | 0       |
| Shine                        | 13     | 0    | 60    | 2       |
| Sunrise                      | 0      | 1    | 0     | 99      |

#### Main diagonal (90, 72, 60, 99) → True Positives 
+ 90 Cloudy photos were correctly predicted.
+ 72 Rain photos were correctly.
+ 60 Shine photos were correctly predicted.
+ 99 Sunrise photos were correctly predicted.

#### Values off the main diagonal → False Positives 
+ 1 Cloudy photos were mistaken for Shine.
+ 13 photos of Shine were mistaken for Cloudy.
+ 2 photos of Shine were mistaken for Sunrise
+ 1 Sunrise photos were mistaken for Shine.

#### Evaluate the model
+ The correct prediction rate is high, especially with Rain (100% accurate) and Sunrise (~99% accurate).
+ The Shine class has the most errors: 13 Shine photos mistaken for Cloudy → Maybe because Shine photos are sometimes cloudy.
+ 2 Shine images were mistaken for Sunrise → Maybe the lighting may be the same between these two layers.
+ A photo of Sunrise was mistaken for Rain → Maybe due to lighting conditions and unclear photo background.



