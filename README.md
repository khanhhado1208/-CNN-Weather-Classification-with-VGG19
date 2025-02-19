# CNN Weather Classification with VGG19
## Multi-class Weather Classification using Transfer Learning (VGG19). Implemented a CNN model using VGG19 to classify weather conditions (Cloudy, Rain, Shine, Sunrise).

### 📌 Introduction
In this exercise, we use the VGG19 model with transfer learning to identify 4 types of weather based on images: 
☁️ Cloudy
🌧 Rain
☀ Shine
🌄 Sunrise

### Objective: 
Two different approaches were tested:

Trainable = False (only Fully Connected Layers trained)

Trainable = True (entire VGG19 retrained)

🎯 Key Findings
✅ Trainable = False → Higher Accuracy (~95%)
✅ Trainable = True → Overfitting, Poor Generalization (~29%)

### Dataset
Source: Multi-class Weather Dataset [https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset]

Number of photos: 1125 photos in 4 classes

Image size: 224x224 pixels

Preprocessing:

Normalize data: Divide pixel value by 255

One-hot encoding for labels

Split the data set: 70% training, 30% validation

🏗️ Model Architecture
#### CNN and set all parameters to trainable = True

<img width="319" alt="image" src="https://github.com/user-attachments/assets/97e8fe37-8c19-41b5-9e5a-359413df0fa7" />

<img width="284" alt="image" src="https://github.com/user-attachments/assets/664c287b-b0e7-4452-a4b3-cccd18c7656a" />


#### Train vs Validation Accuracy
+ Trend: Accuracy increases gradually with the number of epochs
+ Low train accuracy (~32%): The model is learning badly on the training set
+ Low validation accuracy (~30%): The model generates badly
+ Train-validation distance: The gap is not too far; but the training and validation scores are very bad, which illustrates the model does not generate well

#### Train vs Validation Loss
+ Trend: Loss of both training set and validation set are gradually decreasing.
+ Loss distance: The gap is quite close to each other, which shows the model is not overfitting.
+ Loss reduction: There are no large fluctuations; providing that training is stable

#### CNN and set all parameters to trainable = False

<img width="294" alt="image" src="https://github.com/user-attachments/assets/4d2719e4-2532-414d-a276-c80ea62f5402" />

<img width="281" alt="image" src="https://github.com/user-attachments/assets/742167d3-7c6e-48c9-b092-ecfff4e6a4b4" />

#### Train vs Validation Accuracy
+ Trend: Accuracy increases gradually with the number of epochs
+ High train accuracy (~90%): The model is learning well on the training set
+ High validation accuracy (~80%): The model generates well
+ Train-validation distance: The gap is not too far (10%); it illustrates no signs of serious overfitting  

#### Train vs Validation Loss
+ Trend: Loss of both training set and validation set are gradually decreasing.
+ Loss distance: The gap is quite close to each other, which shows the model is not overfitting.
+ Loss reduction: There are no large fluctuations; providing that training is stable.


📊 Model Performance
#### Confusion Matrix (Trainable = False)

<img width="275" alt="image" src="https://github.com/user-attachments/assets/bf4fd005-9afe-436e-b4e5-c4824e3f6bd1" />


| True Label / Predicted label | Cloudy | Rain | Shine | Sunrise |
|------------------------------|--------|------|-------|---------|
| Cloudy                       | 0      | 0    | 0     | 91      |
| Rain                         | 0      | 0    | 0     | 72      |
| Shine                        | 0      | 0    | 0     | 75      |
| Sunrise                      | 0      | 0    | 0     | 100     |

#### Main diagonal (0, 0, 0, 100) → True Positives 
+ 0 Cloudy photos were correctly predicted.
+ 0 Rain photos were correctly.
+ 0 Shine photos were correctly predicted.
+ 100 Sunrise photos were correctly predicted (perfect accuracy for Sunrise class).
+ This means that only the "Sunrise" class is recognized correctly, all other classes are confused.

#### Values out off the main diagonal → False Positives 
+ 91 Cloudy photos were mistaken for Sunrise.
+ 72 photos of Rain were mistaken for Sunrise.
+ 75 photos of Shine were mistaken for Sunrise.
+ This shows that the model is tending to predict all images as "Sunrise".

#### Evaluate the model
+ Poor classification performance overall.
+  Cloudy, Rain, Shine have 0% accuracy.
+  Extreme class bias toward Sunrise.

#### Confusion Matrix (Trainable = True)

<img width="254" alt="image" src="https://github.com/user-attachments/assets/c38bfb15-52ff-445b-9d89-3b59ea085d36" />

| True Label / Predicted label | Cloudy | Rain | Shine | Sunrise |
|------------------------------|--------|------|-------|---------|
| Cloudy                       | 69     | 21   | 1     | 0       |
| Rain                         | 0      | 72   | 0     | 0       |
| Shine                        | 7      | 24   | 42    | 2       |
| Sunrise                      | 5      | 8    | 7     | 80      |


#### Main diagonal (60, 72, 42, 80) → True Positives 
+ 69 Cloudy photos were correctly predicted.
+ 72 Rain photos were correctly.
+ 42 Shine photos were correctly predicted.
+ 80 Sunrise photos were correctly predicted.

#### Values out off the main diagonal → False Positives 
+ 21 Cloudy photos were mistaken for Rain.
+ 7 photos of Shine were mistaken for Cloudy.
+ 24 photos of Shine were mistaken for Rain.
+ 2 photos of Shine were mistaken for Sunrise.
+ 5 photos of Sunrise were mistaken for Cloudy.
+ 8 photos of Sunrise were mistaken for Rain.
+ 7 photos of Sunrise were mistaken for Shine.

#### Evaluate the model
+ Shine and Cloudy had the most confusion with other classes.
+ Sunrise had the highest prediction accuracy (80%).


### Results & Reviews

| Method            | Train Accuracy | Validation Accuracy | Overfitting             |
|-------------------|----------------|---------------------|-------------------------|
| Trainable = True  | 98.83%         | 29.59%              | ❌ High Overfitting      |
| Trainable = False | 95.27%         | 93.49%              | ✅ Better Generalization |

📌 Key Insight:

Training all layers in VGG19 hurts performance on a small dataset.
Freezing earlier layers prevents weight overwriting and improves test accuracy.

### Confusion Matrix

📌 Conclusion:
🔹 Freezing most VGG19 layers while training only the Fully Connected layers gives better results.
🔹 Retraining the entire model leads to performance degradation due to the small dataset.

| Train Method      | When to be used?                                                                                             | 
|-------------------|--------------------------------------------------------------------------------------------------------------|
| Trainable = True  | When there is a large data set, it helps the model learn more optimally. Needs a more powerful GPU to train. |  
| Trainable = False | When the dataset is small, avoid overfitting but may suffer from underfitting. Train faster.                 |   

### 📌[Project Documentation](http://ha-robo.com/blog/5)
