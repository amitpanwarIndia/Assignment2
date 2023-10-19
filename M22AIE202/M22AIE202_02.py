import tensorflow as tf
#from tensorflow import stl10
#from tensorflow import ImageDataGenerator
#from tensorflow.keras.datasets import stl10
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
import resource
import tensorflow_datasets as tfds

def create_autoencoder(target_size, bottleneck_dim):    
    input_img = keras.layers.Input(shape=target_size)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoded = keras.layers.Conv2D(bottleneck_dim, (3, 3), activation='relu', padding='same')(x)
    encoder = keras.Model(input_img, encoded)
    
    # Compile the model
    pretrained_encoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return encoder

#download dataset
#(x_train, y_train), (x_test, y_test) = stl10.load_data()
#ds = tfds.load('stl10', split=[]'train', shuffle_files=True)

(x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load(
    'stl10',
    split=['train', 'test'],
    batch_size=-1,
    as_supervised=True,
))

# Preprocess images
x_train = x_train / 255.0 
x_test = x_test / 255.0

# Loading pre-trained encoder
pretrained_encoder = tf.keras.models.load_model('encoder_weights.h5')

# Extract features
x_train_features = pretrained_encoder.predict(x_train)
x_test_features = pretrained_encoder.predict(x_test)


# Define the MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(256,)), 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_features, y_train, epochs=10, batch_size=32)

model.fit(x_test_features, y_test, epochs=5, batch_size=32)

test_loss, test_accuracy = model.evaluate(x_test_features, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

y_pred = model.predict(x_test_features)
y_pred_labels = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix:")
print(cm)

fpr, tpr, _ = roc_curve(y_test, y_pred[:, 0])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()



#Different model for implementation

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(32, 32, 3)),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])


# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=10, batch_size=32)
