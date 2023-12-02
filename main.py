import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)


loaded_model = tf.keras.models.load_model('car_logo_classifier_model.h5')


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array


image_path = input("Enter the path of the image: ")


processed_image = preprocess_image(image_path)

plt.figure(figsize=(10, 4))

# original Image
plt.subplot(1, 2, 1)
img = image.load_img(image_path)
plt.imshow(img)
plt.title('Original Image')

# preprocessed Image
plt.subplot(1, 2, 2)
plt.imshow(np.squeeze(processed_image))
plt.title('Preprocessed Image')

# plt.show()

# make predictions
predictions = loaded_model.predict(processed_image)
# print(predictions[0])

# get the predicted class
predicted_class = np.argmax(predictions)

# class_labels = list(train_generator.class_indices.keys())
# print(class_labels)

class_labels = ['Buick', 'Chery', 'Citroen', 'Honda', 'Hyundai', 'Lexus', 'Mazda', 'Peugeot', 'Toyota', 'VW']

print(f"PREDICTION => : {class_labels[predicted_class]}")