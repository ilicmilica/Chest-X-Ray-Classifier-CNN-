import tensorflow as tf 
import os
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import cv2
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras import layers, models
from sklearn.utils import shuffle

# Path to the folders that contain train and test datasets
data_train = "C:/Programming/data/chest_xray/train"
data_test = "C:/Programming/data/chest_xray/test" 

# Upload the data and resize the images
batch_size = 32
img_size = (224, 224)

train_data = tf.keras.utils.image_dataset_from_directory(
    data_train,
    #validation_split=0.2,
    #subset="training",
    #seed=42,
    image_size=img_size,
    batch_size=batch_size
)

test_data = tf.keras.utils.image_dataset_from_directory(
    data_train,
    #validation_split=0.2,
    #subset="validation",
    #seed=42,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_data.class_names
print(class_names)  # ['normal', 'pneumonia']

# limit data to X% of the dataset
train_data = train_data.take(100)


# build the model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),  # Normalize images
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(train_data, epochs=6, callbacks=[callback])

# Save the model as .h5 file
model.save("model/model.keras")

# Load the model
model = tf.keras.models.load_model("model/model.keras")

# Test the model
test_loss, test_acc = model.evaluate(test_data)
print("Test accuracy", test_acc)
print("Test loss", test_loss)

#______________________________
y_pred = []
y_true = []

for images, labels in test_data:
    preds = model.predict(images)
    y_pred.extend(tf.argmax(preds, axis=1).numpy())
    y_true.extend(labels.numpy())

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=class_names))
