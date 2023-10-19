import keras
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize(target_size[:2])
    image = np.array(image) / 255.0
    return image

def split_dataset(images):
    X_train, X_temp = train_test_split(images, test_size=1 - split_ratio, random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=0.67, random_state=42)
    return X_train, X_val, X_test

def load_dataset(dataset_dir):
    preprocessed_images = []
    count=0
    for image_file in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, image_file)
        preprocessed_images.append(preprocess_image(image_path))
        count = count + 1
        if count > 2000:
            break
    return preprocessed_images

def create_autoencoder(target_size, bottleneck_dim, learning_rate=0.01):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    input_img = keras.layers.Input(shape=target_size)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    encoded = keras.layers.Conv2D(bottleneck_dim, (3, 3), activation='relu', padding='same')(x)

    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)    
    autoencoder = keras.Model(input_img, decoded)
    encoder = keras.Model(input_img, encoded)
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])
    return autoencoder, encoder

def train(num_epochs, batch_size, train_data, val_data):
    autoencoder.fit(train_data, train_data,
                epochs=num_epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(val_data, val_data))

def create_mask(image, mask_percentage):
        mask = np.random.rand(*image.shape) < mask_percentage / 100.0
        masked_image = image.copy()
        masked_image[mask] = 0
        return masked_image

def evaluate_autoencoder(test_data):
    reconstructed_images = autoencoder.predict(test_data)
    mse_errors = []
    i=0
    for i in range(len(test_data)):
        mse = mean_squared_error(test_data[i].reshape(-1), reconstructed_images[i].reshape(-1))
        mse_errors.append(mse)
        
    mse = mean_squared_error(test_data.reshape(test_data.shape[0], -1), reconstructed_images.reshape(reconstructed_images.shape[0], -1))
    mae = mean_absolute_error(test_data.reshape(test_data.shape[0], -1), reconstructed_images.reshape(reconstructed_images.shape[0], -1))    

    psnr_value = calculate_psnr(test_data[i], reconstructed_images[i])
    print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr_value:.4f}')

    return mse,mae,mse_errors

def plot_image(test_images, decoded_images):
    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        plt.subplot(131)
        plt.imshow(test_images[i].reshape(128, 128, 3))

        plt.subplot(132)
        plt.imshow(decoded_images[i].reshape(128, 128, 3))
    plt.show()

def calculate_psnr(original, reconstructed):
        # Ensure that the images are in the range [0, 1]
        original = np.clip(original, 0, 1)
        reconstructed = np.clip(reconstructed, 0, 1)
        
        # Calculate PSNR
        psnr_value = peak_signal_noise_ratio(original, reconstructed)
        return psnr_value

def noise(images):
    noise_factor = 0.1
    mean = 0
    var = 10
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, size=images.shape) 

    noisy_array = images + noise_factor * gaussian
    return np.clip(noisy_array, 0.0, 1.0)

#parameters
target_size=(128, 128, 3)

#split_ratio = .8
split_ratio = .7
bottleneck_dim = 256
mask_percentage=20
epochs=20
batch_size=128

dataset_dir = "C:\M.Tech.Stuff\DeepLearning\Assignment\Second\VOCtrainval_06_Nov_2007\VOCdevkit\VOC2007\JPEGImages\\"
images = load_dataset(dataset_dir)

x_train, x_val, x_test = split_dataset(images)
x_train = np.reshape(x_train, (len(x_train), 128, 128, 3))
x_val = np.reshape(x_val, (len(x_val), 128, 128, 3))
x_test = np.reshape(x_test, (len(x_test), 128, 128, 3))

noisy_train_data = noise(x_train)
noisy_val_data = noise(x_val)
noisy_test_data = noise(x_test)

print(noisy_train_data.shape)
print(noisy_val_data.shape)
print(noisy_test_data.shape)

autoencoder,_= create_autoencoder(target_size, bottleneck_dim)

train(epochs,batch_size, noisy_train_data, noisy_val_data)

#encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(noisy_test_data)

mse,mae,mse_errors = evaluate_autoencoder(noisy_test_data)
print(f"MSE: {mse}, MAE: {mae}")

# Plot the reconstruction errors
plt.figure(figsize=(8, 4))
plt.hist(mse_errors, bins=50)
plt.title('Reconstruction Error Histogram')
plt.xlabel('Mean Squared Error')
plt.ylabel('Frequency')
plt.show()

plot_image(noisy_test_data,decoded_imgs)