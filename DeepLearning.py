#Import working packages
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.layers import Softmax
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, classification_report

#images folder path
data_dir = r"C:/Users/rania/Downloads/curated_data"
#List the folder of path ---> contain a train and test folders
print(os.listdir(data_dir))

#train folder contain 3 classes
train_dir = data_dir+"/train"
print(os.listdir(train_dir))

# Total images for each training class
print("Nombre des images de de personnes normales dans le training set : {}" .format(len(os.listdir(train_dir+"/1NonCOVID"))))
print("Nombre des images de patients atteints de la COVID 19 dans le training set : {}" .format(len(os.listdir(train_dir+"/2COVID"))))

#test folder contain 3 classes
test_dir = data_dir+"/test"
print(os.listdir(train_dir))

print("Nombre des images de de personnes normales dans le training set : {}" .format(len(os.listdir(test_dir+"/1NonCOVID"))))
print("Nombre des images de patients atteints de la COVID 19 dans le training set : {}" .format(len(os.listdir(test_dir+"/2COVID"))))

covid = imread(train_dir+"/2COVID"+"/14_Jun_coronacases_case7_75.png")
normal = imread(train_dir+"/1NonCOVID"+"/16_Morozov_study_0017_20.png")

plt.figure(figsize=(10,20))
plt.subplot(1, 2, 1)
plt.imshow(covid, cmap="gray")
plt.title('Covid19 patient X-Ray',fontsize= 10)
plt.subplot(1, 2, 2)
plt.imshow(normal, cmap="gray")
plt.title('Normal person X-Ray',fontsize= 10)
plt.show()

#Create images generator with normalization
generator = ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=0.3, width_shift_range=0.0,
    height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
    channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
    horizontal_flip=False,  vertical_flip=False,rescale=None,
    preprocessing_function=None, data_format="channels_last", validation_split=0.0, dtype=None
)

#train images generator
average_image_size = (512,512,1)
train_generator = generator.flow_from_directory (
    train_dir,
    target_size=average_image_size[:2],
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=60,
    shuffle=True,
)

#test images generator
test_generator = generator.flow_from_directory (
    test_dir,
    target_size=average_image_size[:2],
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=60,
    shuffle=False,
)

#stop training if val_loss is not dropping after six successif try
stop = EarlyStopping(monitor="val_loss", mode="min", patience=6)

#Create 6layers CNN Model
model = Sequential()
model.add(Conv2D(filters = 32, padding = "same", kernel_size = (2,2), strides = (2,2), activation = "relu", input_shape = average_image_size, kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001)))
model.add(MaxPool2D(2))
model.add(Dropout(.2))
model.add(Conv2D(filters = 64, padding = "same", kernel_size = (2,2), strides = (2,2), activation = "relu", kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001) ))
model.add(MaxPool2D(2))
model.add(Dropout(.2))
model.add(Conv2D(filters = 128, padding = "same", kernel_size = (2,2), strides = (2,2), activation = "relu", kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001)))
model.add(MaxPool2D(2))
model.add(Dropout(.2))
model.add(Conv2D(filters = 128, padding = "same", kernel_size = (2,2), strides = (2,2), activation = "relu", kernel_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001)))
model.add(MaxPool2D(2))
model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(units = 64, activation = "relu"))
model.add(Dense(units = 2, activation = "softmax"))
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

print("model.summary()")

model.fit(train_generator, validation_data=test_generator, epochs=50)

print(model.metrics_names)

pd.DataFrame(model.history.history)[["loss", "val_loss"]].plot(figsize =(16,6), marker = "o", mfc = "r")
pd.DataFrame(model.history.history)[["accuracy", "val_accuracy"]].plot(figsize =(16,6), marker = "o", mfc = "r")

predictions = model.predict(test_generator)

pred_labels = np.argmax(predictions, axis = 1)
len(test_generator.classes)
len(pred_labels)


print(classification_report(test_generator.classes, pred_labels))
print(confusion_matrix(test_generator.classes, pred_labels))

model.evaluate(test_generator)

fpr, tpr, _ = roc_curve(test_generator.classes, pred_labels, pos_label=1)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC CURVE FOR LABEL COVID")
plt.legend(loc="lower right")
plt.show()
