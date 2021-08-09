import wx
# More imports
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, \
  preprocess_input
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
import tensorflow as tf
print(tf.__version__)
from glob import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
def onButton(event):
    print("Button pressed.")
psxxo=wx.App()
frame=wx.Frame(None, -1, 'training')
pa = wx.Panel(frame)
# Create text input
dlg = wx.TextEntryDialog(pa, 'Enter the training dataset path','training')
dlg2 = wx.TextEntryDialog(pa, 'Enter the validation dataset path','training')
dlg.SetValue("")
dlg2.SetValue("")

cwd = os.getcwd()
if dlg.ShowModal()==wx.ID_OK :

    if dlg2.ShowModal()==wx.ID_OK:
        train_path = dlg.GetValue()
        valid_path = dlg2.GetValue()
        IMAGE_SIZE = [200, 200]
        # useful for getting number of files
        image_files = glob(train_path + '/*/*.jpg')
        valid_image_files = glob(valid_path + '/*/*.jpg')
        folders = glob(train_path + '/*')
        batch_size = 128

        # create an instance of ImageDataGenerator
        gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        # create generators
        train_generator = gen.flow_from_directory(
        train_path,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        )

        valid_generator = gen.flow_from_directory(
        valid_path,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='binary',
        )

        IMG_WIDTH=200
        IMG_HEIGHT=200
        def create_dataset(img_folder):
            img_data_array=[]
            class_name=[]
        
            for dir1 in os.listdir(img_folder):
                for file in os.listdir(os.path.join(img_folder, dir1)):
            
                    image_path= os.path.join(img_folder, dir1,  file)
                    image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
                    image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
                    image=np.array(image)
                    image = image.astype('float32')
                    image /= 255 
                    img_data_array.append(image)
                    class_name.append(dir1)
            return img_data_array, class_name
        # extract the image array and class name
        train_data, train_class =create_dataset(train_path)
        val_data, val_class = create_dataset(valid_path)
        def target(class_name):
            target_dict={k: v for v, k in enumerate(np.unique(class_name))}
            target_dict
            target_val =  [target_dict[class_name[i]] for i in range(len(class_name))]
            return target_val
        train_tar = target(train_class)
        valid_tar = target(val_class)
        K = len(train_class)
        # Build the model using the functional API
        i = Input(shape=train_data[0].shape)
        # x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
        # x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
        # x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        # x = Dropout(0.2)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        # x = Dropout(0.2)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        # x = Dropout(0.2)(x)

        # x = GlobalMaxPooling2D()(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(K, activation='softmax')(x)

        model = Model(i, x)
        # Compile
    # Note: make sure you are using the GPU for this!
        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
                # Fit
        r = model.fit(x=np.array(train_data, np.float32), y=np.array(list(map(int,train_tar)), np.float32), validation_data=(np.array(val_data, np.float32), np.array(list(map(int,valid_tar)), np.float32)), epochs=30)
        model.save('model.h5')
        wx.MessageBox("success", 'status', wx.OK)

    
    
dlg.Destroy()