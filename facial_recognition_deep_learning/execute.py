#!/usr/bin/env python
# coding: utf-8

# # 1. Install Dependencies and Setup

# In[1]:


# get_ipython().system('pip install tensorflow opencv-python matplotlib')


# In[2]:


# get_ipython().system('pip list')


# In[ ]:


import tensorflow as tf
import os


# In[2]:


# Avoid OOM errors by setting GPU Memory Consumption Growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus: 
#     tf.config.experimental.set_memory_growth(gpu, True)


# In[3]:


# tf.config.list_physical_devices('GPU')


# # 2. Remove dodgy images

# In[4]:


import cv2
import imghdr


# # In[5]:


data_dir = 'data' 


# # In[6]:


image_exts = ['jpeg','jpg']


# # In[7]:


# for image_class in os.listdir(data_dir): 
#     if(image_class == '.DS_Store'):
#         continue
#     for image in os.listdir(os.path.join(data_dir, image_class)):
#         image_path = os.path.join(data_dir, image_class, image)
#         print(image_path)
#         try: 
#             img = cv2.imread(image_path)
#             tip = imghdr.what(image_path)
#             if tip not in image_exts: 
#                 print('Image not in ext list {}'.format(image_path))
#                 os.remove(image_path)
#         except Exception as e: 
#             print('Issue with image {}'.format(image_path))
#             os.remove(image_path)


# # # 3. Load Data

# # In[8]:


import numpy as np
from matplotlib import pyplot as plt


# # In[9]:

# class_labels = ['Achale Ebot', 'Agyingi Jan', 'Ataba Emmanuel', 'Bebongchu Yannick', 'Besingi Naura', 'Egbeakwene Denning', 'Frank', 'Kono Steve', 'Njie Etah', 'Randy Kwalar']
# data = tf.keras.utils.image_dataset_from_directory('data', class_names=class_labels)


# # In[10]:


# data_iterator = data.as_numpy_iterator()


# # In[11]:


# batch = data_iterator.next()
# batch[0].shape
# print(batch[1])

# # In[12]:


# fig, ax = plt.subplots(ncols=10, figsize=(20,20))
# for idx, img in enumerate(batch[0][:10]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])

# plt.show()

#Achale = 0
#Jan = 1
#Ataba = 2
#Bebongchu = 3
#naura = 4
# denning = 5
# frank = 6
# kono = 7
# Etah = 8
# Randy = 9


# # # 4. Scale Data

# In[13]:


# data = data.map(lambda x,y: (x/255, y))


# # In[ ]:


# data.as_numpy_iterator().next()


# # # 5. Split Data

# # In[15]:

# print(len(data))
# train_size = int(len(data)*.6)
# val_size = int(len(data)*.2)+3
# test_size = int(len(data)*.1)+2


# print(train_size)
# print(val_size)
# print(test_size)

# # In[16]:


# train_size


# # In[17]:


# train = data.take(train_size)
# val = data.skip(train_size).take(val_size)
# test = data.skip(train_size+val_size).take(test_size)


# # # 6. Build Deep Learning Model

# # In[18]:


# train


# # In[19]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# # In[20]:


# model = Sequential()


# # In[21]:


# model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (3,3), 1, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(16, (3,3), 1, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# # model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(11, activation='softmax'))
# 

# # In[22]:


# model.compile('adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])


# # In[23]:


# model.summary()


# # # 7. Train

# # In[24]:


logdir='logs'


# # In[25]:


# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# # In[ ]:


# hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


# # # 8. Plot Performance

# # In[27]:


# fig = plt.figure()
# plt.plot(hist.history['loss'], color='teal', label='loss')
# # plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
# fig.suptitle('Loss', fontsize=20)
# plt.legend(loc="upper left")
# plt.show()


# # In[28]:


# fig = plt.figure()
# plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
# # plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
# fig.suptitle('Accuracy', fontsize=20)
# plt.legend(loc="upper left")
# plt.show()


# # # 9. Evaluate

# # In[29]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# # In[30]:


# pre = Precision()
# re = Recall()
# acc = BinaryAccuracy()


# # In[31]:


# for batch in test.as_numpy_iterator(): 
#     X, y = batch
#     yhat = model.predict(X)
#     y_pred = tf.argmax(yhat, axis=1)
#     pre.update_state(y, y_pred)
#     re.update_state(y, y_pred)
#     acc.update_state(y, y_pred)




# # In[32]:


# print(pre.result(), re.result(), acc.result())


# # # 10. Test

# # In[33]:


import cv2
from tensorflow.keras.models import load_model


# # In[39]:


# img = cv2.imread('pic.jpg')
# plt.imshow(img)
# plt.show()


# # In[40]:


# resize = tf.image.resize(img, (256,256))
# plt.imshow(resize.numpy().astype(int))
# plt.show()


# # In[41]:


# new_model = load_model(os.path.join('models','model.h5'))
# yhat = new_model.predict(np.expand_dims(resize/255, 0))


# # In[42]:


# yhat


# # In[43]:

# print(yhat)
# if np.any(yhat > 0.5): 
#     print(f'Predicted class is Agyingi')
# else:
#     print(f'Predicted class is Is not Agyingi')


# # # 11. Save the Model

# # In[44]:


# from tensorflow.keras.models import load_model


# # In[45]:


# model.save(os.path.join('models','model.keras'))
# model.save(os.path.join('models','model.h5'))


# # In[46]:


# new_model = load_model(os.path.join('models','model.h5'))


# # In[47]:


# new_model.predict(np.expand_dims(resize/255, 0))

# from tensorflow. import lite
model = tf.keras.models.load_model(os.path.join('models','model.h5'))
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("models/new_model.tflite", "wb").write(tflite_model)
