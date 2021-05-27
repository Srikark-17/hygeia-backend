import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import * 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, accuracy_score

#set seed
from numpy.random import seed
seed(10)
tf.random.set_seed(10)

print('Imported Successfully')

img_folder='./dataset/xray_dataset_covid19/train/'
plt.figure(figsize=(20,20))
for i in range(6):
 class_ = random.choice(os.listdir(img_folder))
 class_path= os.path.join(img_folder, class_)
 file=random.choice(os.listdir(class_path))
 image_path= os.path.join(class_path,file)
 print(image_path)
 img= mpimg.imread(image_path)
 ax=plt.subplot(1,6,(i+1))
 plt.imshow(img)
 ax.title.set_text(class_)

def create_dataset(img_folders,IMG_WIDTH,IMG_HEIGHT):
    
    
    img_data_array=[]
    class_name=[]
    n=0
    for dirname, _, filenames in os.walk(img_folders):
        for filename in filenames:
            img_path= os.path.join(dirname, filename)
            #read the image
            image = cv2.imread(img_path)
            # BGR is converted to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_WIDTH,IMG_HEIGHT))
            image=np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_=str(dirname).split("/")[-1]
            class_name.append(class_)
            n+=1
            
            
            
    return img_data_array, class_name,n# extract the image array and class name

IMG_WIDTH= 224
IMG_HEIGHT=224
train_path='./dataset/xray_dataset_covid19/train/'
test_path='./dataset/xray_dataset_covid19/test/'
train_img,train_target,num_img=create_dataset(train_path,IMG_WIDTH,IMG_HEIGHT)
test_img,test_target,num_test_img=create_dataset(test_path,IMG_WIDTH,IMG_HEIGHT)

plt.figure(figsize=(20,20))
for i in range(6):
    random_num = random.randint(0,num_img)
    ax=plt.subplot(1,6,(i+1))
    plt.imshow(train_img[random_num])
    ax.title.set_text(train_target[random_num])

target_dict={k: v for v, k in enumerate(np.unique(train_target))}
print(target_dict)
train_target= [target_dict[train_target[i]] for i in range(len(train_target))]
train_target=np.array(train_target)
train_img=np.array(train_img)
test_target= [target_dict[test_target[i]] for i in range(len(test_target))]
test_target=np.array(test_target)
test_img=np.array(test_img)

#Saving best model while monitoring accuracy
model_chkpt = ModelCheckpoint('covid-model.h5', save_best_only=True, monitor='accuracy')

#early stopping for preventing overfitting
early_stopping = EarlyStopping(monitor='loss', restore_best_weights=False, patience=10)

#Define a Sequential() model.
model = Sequential()
# Add the first layer: 32 is the number of filters; kernel_size specifies the size of our filters;
# activation specifies the activation function;input_shape specifies what type of input we are going to pass to the network
model.add(Conv2D(32, kernel_size=(3,3), activation="relu",input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
# Second layer: specified 64 filters(must be a power of 2)
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
#Deine a Max pooling: kernel_size, which specified the size of the pooling window.
model.add(MaxPooling2D(pool_size=(2,2)))
#Dropout. This means that the model will not overfit, as some neurons randomly will not be selected for activation. 
#This prevents the model from overfitting.
model.add(Dropout(0.25))
#Repeate the above steps to make a deeper network.
model.add(Conv2D(128, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Flatten layer
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))
# Create an output sigmoid function
model.add(Dense(1, activation="sigmoid"))
#Compile the model: binary_crossentropy because this is a binary classification problem; adam as the optimizer; the metric that we want to monitor is accuracy.
model.compile(loss="binary_crossentropy", optimizer="adam",metrics = ["accuracy"])
# Printe the model architecture to take a look at the number of parameters that the model will learn.
model.summary()

history = model.fit(train_img, train_target, 
          validation_split=0.10, 
          epochs=10, 
          batch_size=32, 
          shuffle=True, 
          callbacks=[model_chkpt, early_stopping]
        )

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()

#prediction on test set
pred = model.predict(test_img,batch_size=32)

label = [int(p>=0.5) for p in pred]

#label
#Performance Evaluation - Accuracy, Classification Report & Confusion Matrix
#Accuracy Score
print ('Accuracy Score : ', accuracy_score(label, test_target), '\n')

#precision, recall report
print ('Classification Report :\n\n' ,classification_report(label, test_target))
