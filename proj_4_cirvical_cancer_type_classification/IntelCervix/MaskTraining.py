from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import CommonModules as CM

base_folder = "H:/Kaggle/preprocessed/train"
mask_folder = "H:/Kaggle/masking_preprocessed/train"


## this is the size of our encoded representations
#encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

input_img = Input(shape=(240, 320, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x2 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)#注意这个地方如果没有padding,默认是valid所以图像大小会发生变化
x = UpSampling2D((2, 2))(x2)
decoded = Conv2D(3, (2, 2), activation='sigmoid', padding='same')(x)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, x2)

# create a placeholder for an encoded (32-dimensional) input
#encoded_input = Input(shape=(encoding_dim,))
#encoded_input = Input(shape=(4, 4, 8))
## retrieve the last layer of the autoencoder model
#decoder_layer = autoencoder.layers[-1]
## create the decoder model
#decoder = Model(encoded_input, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# we create two instances with the same arguments
data_gen_args = dict(rotation_range=90.,
                     width_shift_range=0.1,            
                     rescale=1./255,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 4

base_folder_data = np.array([np.array(Image.open(fname).resize((320,240))) for fname in CM.IO.ListFiles(base_folder, ".jpg", All = True)]) #注意这个地方顺序是(320, 240), 非常奇怪
mask_folder_data = np.array([np.array(Image.open(fname).resize((320,240))) for fname in CM.IO.ListFiles(mask_folder, ".jpg", All = True)])
print(base_folder_data.shape)
print(mask_folder_data.shape)

image_generator = image_datagen.flow(
    base_folder_data,
    #class_mode=None,
    #target_size=(240, 320),
    batch_size = 16,
    seed=seed)

mask_generator = mask_datagen.flow(
    mask_folder_data,
    #class_mode=None,
    #target_size=(240, 320),
    batch_size = 16,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

autoencoder.load_weights("best_model_mask_2.h5")

autoencoder.fit_generator(
    train_generator,
    steps_per_epoch=500,
    epochs=200,
    validation_data=train_generator,
    validation_steps=50,    
    callbacks = [TensorBoard(log_dir='./log'), ModelCheckpoint("best_model_mask_3.h5", verbose = 1, save_best_only = True, save_weights_only = True)])

#for epoch in range(200):
#    print('Epoch', epoch)
#    batches = 0
#    for x_batch, y_batch in train_generator:
#        autoencoder.fit(x_batch, y_batch)
#        batches += 1
#        if batches >= len(x_train) / 32:
#            # we need to break the loop by hand because
#            # the generator loops indefinitely
#            break

autoencoder.save_weights('first_try_mask.h5')

#autoencoder.fit(x_train, x_train,
#                epochs=50,
#                batch_size=256,
#                shuffle=True,
#                validation_data=(x_test, x_test),
#                callbacks=[TensorBoard(log_dir='./log')])

# encode and decode some digits
# note that we take them from the *test* set
x_test = image_generator.next() #一次产生16个(batch size那么多的)
y_test = mask_generator.next()
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)
print(x_test.shape, y_test.shape, encoded_imgs.shape, decoded_imgs.shape)

n = 5  # how many digits we will display
plt.figure(figsize=(20, 4), tight_layout = True)
for i in range(n):
    #print(encoded_imgs[i])
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(240, 320, 3))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(240, 320, 3))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.tight_layout(pad = 0.1)
plt.show()