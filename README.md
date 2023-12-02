# Car Logo Classifier Model

## 1. Introduction

Welcome to the Car Logo Classifier Model! This model is built using TensorFlow and Convolutional Neural Networks (CNN) to classify logos of 10 different car brands. The supported logos are: ['Buick', 'Chery', 'Citroen', 'Honda', 'Hyundai', 'Lexus', 'Mazda', 'Peugeot', 'Toyota', 'VW'].

## 2. Usage

### Training

To train the model, follow these steps:

1. Run `pip install -r requirements.txt` to install the required dependencies.
2. Open `logos.ipynb` file and run each section. It will generate the model file `car_logo_classifier_model.h5`.

### Running the Model

To use the pre-trained model without training, follow these steps:

1. Install the requirements by running `pip install -r requirements.txt`.
2. Run `python downloadModel.py` to download the pre-trained model to your working directory.
3. Run `python main.py` and input the file path (image path) when prompted. Press Enter, and it will provide you with the prediction.

## 3. About the Model

### Architecture

The model architecture is illustrated below:

```
# Neural Network Architecture

## input Layer
Input

## Convolutional Layer 1
-> Conv2D(32) - ReLU: Convolutional layer with 32 filters of size (3, 3) and ReLU activation.

## MaxPooling Layer 1
-> MaxPooling2D(2x2): Max pooling layer with a pool size of (2, 2).

## Convolutional Layer 2
-> Conv2D(64) - ReLU: Convolutional layer with 64 filters of size (3, 3) and ReLU activation.

## MaxPooling Layer 2
-> MaxPooling2D(2x2): Max pooling layer with a pool size of (2, 2).

## Convolutional Layer 3
-> Conv2D(128) - ReLU: Convolutional layer with 128 filters of size (3, 3) and ReLU activation.

## MaxPooling Layer 3
-> MaxPooling2D(2x2): Max pooling layer with a pool size of (2, 2).

## Flatten Layer
-> Flatten: Reshapes the data before fully connected layers.

## Dense Layer 1
-> Dense(256) - ReLU: Fully connected layer with 256 neurons and ReLU activation.

## Dropout Layer
-? Dropout(0.5): Dropout layer with a dropout rate of 0.5 to prevent overfitting.

## Dense Layer 2 (Output Layer)
-> Dense(10) - Softmax: Fully connected layer with 10 neurons and softmax activation for classification.

## Output Layer
Output

```

### Accuracy

The model boasts an impressive accuracy of 99.45%.

### Graph

![Graph](https://firebasestorage.googleapis.com/v0/b/scrapper-6e7db.appspot.com/o/Screenshot%20at%202023-12-02%2021-23-09.png?alt=media&token=5896a513-4f51-4e7c-876b-874f59eb069f).

### Classes

The model can classify images into 10 classes: ['Buick', 'Chery', 'Citroen', 'Honda', 'Hyundai', 'Lexus', 'Mazda', 'Peugeot', 'Toyota', 'VW'].

### Image Visualization

Uncomment the following code in `logos.ipynb` to view the processed and input images:

```python
# Uncomment the following lines to view the processed and input images
# plt.subplot(1, 2, 1)
# plt.imshow(np.squeeze(original_image))
# plt.title('Input Image')

# preprocessed Image
# plt.subplot(1, 2, 2)
# plt.imshow(np.squeeze(processed_image))
# plt.title('Preprocessed Image')

# plt.show()
```

## 4. Additional Sections

### Dataset

The dataset used for training can be found [here](https://www.kaggle.com/code/binhminhs10/vehicle-logo-recog-10-class).

### Model URL

You can download the pre-trained model directly from [this link](https://firebasestorage.googleapis.com/v0/b/scrapper-6e7db.appspot.com/o/car_logo_classifier_model.h5?alt=media&token=0eb9bf60-ce7c-426c-bcf5-9f0566d79d90).

