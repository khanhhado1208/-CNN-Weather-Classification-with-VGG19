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

# Set trainable (True, False)
isTrainable = True

baseModel = VGG19(input_shape=target_size, weights='imagenet', include_top=False) # Input model use VGG19 use imagenet weight

# Freeze all layers of VGG19
for layer_ctn, layer in enumerate(baseModel.layers[:]):
    layer.trainable = isTrainable
    
# Flatten layer to convert from 4D tensor -> 1D vector
x = Flatten()(baseModel.output)

# Fully Connected Layers (Dense + Dropout)
x = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=0, l2=0.01))(x) # Add a Dense layer with 512 units
x = Dropout(0.1)(x) # Dropout layer with 0.1 unit
x = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0, l2=0.01))(x) # Add a Dense layer with 256 units
x = Dropout(0.1)(x) # Dropout layer with 0.1 unit

# Output layer with 4 classes for the final classifier 
x = Dense(4, activation='softmax')(x)  # 4 classes: ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# Combine base_model and Fully Connected layers into a final model
model = Model(inputs=baseModel.input, outputs=x)

model.summary() # Print mode summary

<img width="248" alt="image" src="https://github.com/user-attachments/assets/295ad89f-6828-44d4-bfa0-bb4e9e87df38" />

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

# Set trainable (True, False)
isTrainable = False

baseModel = VGG19(input_shape=target_size, weights='imagenet', include_top=False) # Input model use VGG19 use imagenet weight

# Freeze all layers of VGG19
for layer_ctn, layer in enumerate(baseModel.layers[:]):
    layer.trainable = isTrainable
    
# Flatten layer to convert from 4D tensor -> 1D vector
x = Flatten()(baseModel.output)

# Fully Connected Layers (Dense + Dropout)
x = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=0, l2=0.01))(x) # Add a Dense layer with 512 units
x = Dropout(0.1)(x) # Dropout layer with 0.1 unit
x = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0, l2=0.01))(x) # Add a Dense layer with 256 units
x = Dropout(0.1)(x) # Dropout layer with 0.1 unit

# Output layer with 4 classes for the final classifier 
x = Dense(4, activation='softmax')(x)  # 4 classes: ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# Combine base_model and Fully Connected layers into a final model
model = Model(inputs=baseModel.input, outputs=x)

model.summary() # Print mode summary

<img width="250" alt="image" src="https://github.com/user-attachments/assets/26d23ade-98b9-4403-a7f9-1fd9a36562f9" />


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

<img width="248" alt="image" src="https://github.com/user-attachments/assets/dc083569-17d8-4f2d-9eea-350e0b82640b" />


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

