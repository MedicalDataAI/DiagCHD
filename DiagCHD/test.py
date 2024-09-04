import os
os.environ['KERAS_BACKEND']='tensorflow'
# import tensorflow as tf
# tf.keras.backend.clear_session()

from keras.losses import CategoricalCrossentropy

import os

import keras
import numpy as np
from keras.applications.xception import Xception
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K

import os

import keras
import numpy as np
from keras.applications.xception import Xception
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adagrad, Adam
import tensorflow as tf





def dice_coeff(x, target, ignore_index = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = tf.shape(x)[0]
    for i in range(batch_size):
        x_i = tf.reshape(x[i], [-1])
        t_i = tf.reshape(target[i], [-1])
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = tf.not_equal(t_i, ignore_index)
            x_i = tf.boolean_mask(x_i, roi_mask)
            t_i = tf.boolean_mask(t_i, roi_mask)
        inter = tf.reduce_sum(tf.multiply(x_i, t_i))
        sets_sum = tf.reduce_sum(x_i) + tf.reduce_sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / tf.cast(batch_size, dtype=tf.float32)


def multiclass_dice_coeff(x, target, ignore_index = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(tf.shape(x)[3]):
        dice += dice_coeff(x[:,:,:,channel], target[:,:,:,channel], ignore_index, epsilon)

    return dice / tf.cast(tf.shape(x)[3], dtype=tf.float32)

def dice_loss(x, target, multiclass = False, ignore_index = -100):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)

def ce_dice_loss(y_true, y_pred):
    ce_loss = CategoricalCrossentropy()(y_true, y_pred)  # 多类别交叉熵损失
    dice_coef = dice_loss(y_pred,y_true,multiclass=True)  # Dice损失

    return ce_loss + dice_coef

def dice_coef(x, target, multiclass = True, ignore_index = -100):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return fn(x, target, ignore_index=ignore_index)




# 数据集的加载
height = 576  #图片的高度
width = 768  #图片的长度
channels = 3  #彩色图片
batch_size = 1  #每一次49个图片
num_classes = 3  #最后是7分类
SEED = 666  # 用于限制多个输入图片的label保持一致，训练集有一个随机扰乱的
epochs = 300
img_size_input_1 = (height, width)  # 第一个输入图片的维度大小
img_size_input_2 = (height, width)  # 第二个输入图片的维度大小
# 数据集加载
train_dir_input_1 = r"E:\zqx\az\CongenitalHeartDisease\Dataset\ClassificationDataset\Train\IMAGE"
valid_dir_input_1 = r"E:\zqx\az\CongenitalHeartDisease\Dataset\ClassificationDataset\Test\IMAGE"
train_dir_input_2 = r"E:\zqx\az\CongenitalHeartDisease\Dataset\ClassificationDataset\Train\MASK"
valid_dir_input_2 = r"E:\zqx\az\CongenitalHeartDisease\Dataset\ClassificationDataset\Test\MASK"

train_datagen_input_1 = keras.preprocessing.image.ImageDataGenerator(  #这个是专门进行图片进行强化操作处理的,传入一个目录就可以将图片转化成你需要的，且进行数据增强，样本会进行叠加的
    rescale=1. / 255, )  #这样就可以新创建出图片了
train_generator_input_1 = train_datagen_input_1.flow_from_directory(
    train_dir_input_1,  #上面的ImageDataGenerator只是一个迭代器，将图片转化成像素值，这个方法flow_from_directory就可以批量取数据
    target_size=img_size_input_1,  #图片大小规定到这个高宽
    batch_size=batch_size,  #每一个批次batch_size个图片进行上面的操作
    seed=SEED,
    shuffle=True,
    class_mode="categorical")  #这个指定二进制标签，我们用了binary_crossentropy损失函数
valid_datagen_input_1 = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)  #验证集不用添加图片，只需要将图片像素值进行规定
valid_generator_input_1 = valid_datagen_input_1.flow_from_directory(
    valid_dir_input_1,
    target_size=img_size_input_1,
    batch_size=batch_size,
    seed=SEED,
    shuffle=False,
    class_mode="categorical")

# mask创建
train_datagen_input_2 = keras.preprocessing.image.ImageDataGenerator()  #这样就可以新创建出图片了
train_generator_input_2 = train_datagen_input_2.flow_from_directory(
    train_dir_input_2,  #上面的ImageDataGenerator只是一个迭代器，将图片转化成像素值，这个方法flow_from_directory就可以批量取数据
    target_size=img_size_input_2,  #图片大小规定到这个高宽
    batch_size=batch_size,  #每一个批次batch_size个图片进行上面的操作
    seed=SEED,
    shuffle=True,
    color_mode='grayscale',
    class_mode="categorical")  #这个指定二进制标签，我们用了binary_crossentropy损失函数
valid_datagen_input_2 = keras.preprocessing.image.ImageDataGenerator()  #验证集不用添加图片，只需要将图片像素值进行规定
valid_generator_input_2 = valid_datagen_input_2.flow_from_directory(
    valid_dir_input_2,
    target_size=img_size_input_2,
    batch_size=batch_size,
    seed=SEED,
    shuffle=False,
    color_mode='grayscale',
    class_mode="categorical")
train_num_input_1 = train_generator_input_1.samples  #获取训练样本总数
train_num_input_2 = train_generator_input_2.samples
valid_num_input_1 = valid_generator_input_1.samples  #获取训练样本总数
valid_num_input_2 = valid_generator_input_2.samples
print("样本总数为：")
print(train_num_input_1, train_num_input_2, valid_num_input_1,
    valid_num_input_2)


# In[ ]:


def generate_data_generator(generator_input_1, generator_input_2):
    while True:
        x_data, label_x = generator_input_1.next()
        mask_data, label_mask = generator_input_2.next()
        # 这一句代码代表输入的多个类型的图片label是一致的
        assert np.array(label_x).all() == np.array(label_mask).all(), '数据集产出失败'
        # 代表输入的图片与输出的label的维度指定
        yield np.array(x_data),{"Seg":tf.one_hot(np.array(tf.squeeze(mask_data,axis=-1)),depth=3),
                                "SegModel_Xception":label_x}


# In[ ]:


from keras.losses import CategoricalCrossentropy
def dice_coeff(x, target, ignore_index = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = tf.shape(x)[0]
    for i in range(batch_size):
        x_i = tf.reshape(x[i], [-1])
        t_i = tf.reshape(target[i], [-1])
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = tf.not_equal(t_i, ignore_index)
            x_i = tf.boolean_mask(x_i, roi_mask)
            t_i = tf.boolean_mask(t_i, roi_mask)
        inter = tf.reduce_sum(tf.multiply(x_i, t_i))
        sets_sum = tf.reduce_sum(x_i) + tf.reduce_sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / tf.cast(batch_size, dtype=tf.float32)


def multiclass_dice_coeff(x, target, ignore_index = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(tf.shape(x)[3]):
        dice += dice_coeff(x[:,:,:,channel], target[:,:,:,channel], ignore_index, epsilon)

    return dice / tf.cast(tf.shape(x)[3], dtype=tf.float32)

def dice_loss(x, target, multiclass = False, ignore_index = -100):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)

def ce_dice_loss(y_true, y_pred):
    ce_loss = CategoricalCrossentropy()(y_true, y_pred)  # 多类别交叉熵损失
    dice_coef = dice_loss(y_pred,y_true,multiclass=True)  # Dice损失

    return ce_loss + dice_coef

def dice_coef(x, target, multiclass = True, ignore_index = -100):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return fn(x, target, ignore_index=ignore_index)

from tensorflow.keras.metrics import categorical_crossentropy


model = load_model(r'callbacks_EarlyStopping.h5',
                   custom_objects={'ce_dice_loss':ce_dice_loss,'dice_coef':dice_coef})


# In[ ]:


test_label = []
test_mask = []
test_pre_label = []
test_pre_mask = []
test_image = []

test_loss = []
test_coef = []
test_iou = []

count = 0
for image,label in generate_data_generator(valid_generator_input_1,valid_generator_input_2):
    mask_label,y_label= label['Seg'],label['SegModel_Xception']
    y_pre_mask,y_pre_label = model.predict(image)
    
    test_loss.append(ce_dice_loss(mask_label,y_pre_mask))
    test_coef.append(dice_coef(y_pre_mask,mask_label))
    test_iou.append(tf.keras.metrics.OneHotMeanIoU(num_classes=3,name='miou')(mask_label,y_pre_mask))
    
    test_label.append(y_label)
    test_pre_label.append(y_pre_label)
    
    test_image.append(image)
    
    test_mask.append(mask_label)
    test_pre_mask.append(y_pre_mask)
    
    count+=1
    if count == 53:
        break

# 百分位法的置信区间
from sklearn import metrics
import numpy as np
# def auc_boot(actual, predicted):
#     # 计算AUC
#     actual = np.array(actual)
#     predicted = np.array(predicted)
#     auc = metrics.accuracy_score(actual, predicted)

#     # 执行bootstrap重采样来计算AUC的置信区间
#     n_bootstraps = 10000
#     auc_scores = []
#     for i in range(n_bootstraps):
#         np.random.seed(i)
#         indices = np.random.choice(range(len(actual)), len(actual), replace=True)
#         # print(indices)
#         auc_bootstrap = metrics.roc_auc_score(actual[indices], predicted[indices])
#         auc_scores.append(auc_bootstrap)
#     # print(auc_scores)
#     low,height = np.percentile(auc_scores, [2.5, 97.5])
#     return auc,low,height

# # print(y_label)
# # print(test_pre_label)
# auc,l,h = auc_boot(test_label,test_pre_label)


# In[ ]:


import random
random.seed=666
aaa = []
aaa.append(random.randint(0,52))
aaa.append(random.randint(0,52))
aaa.append(random.randint(0,52))
aaa.append(random.randint(0,52))


# In[ ]:


# import matplotlib.pyplot as plt
# # 创建一个包含四个子图的图像布局
# fig, axes = plt.subplots(2, 2, figsize=(7, 7))

# # 在每个子图中绘制内容
# axes[0, 0].imshow(np.array(test_image).squeeze()[aaa[0]])  # 第一个子图
# axes[0, 1].imshow(np.array(test_image).squeeze()[aaa[1]])  # 第二个子图
# axes[1, 0].imshow(np.array(test_image).squeeze()[aaa[2]])  # 第一个子图
# axes[1, 1].imshow(np.array(test_image).squeeze()[aaa[3]])  # 第一个子图


# In[ ]:


import matplotlib.pyplot as plt
# 创建一个包含四个子图的图像布局
fig, axes = plt.subplots(2, 2, figsize=(7, 7))

# 在每个子图中绘制内容
axes[0, 0].imshow(np.argmax(np.array(test_mask).squeeze(),axis=-1).reshape((53, 576, 768,1))[aaa[0]],cmap='gray')  # 第一个子图
axes[0, 1].imshow(np.argmax(np.array(test_mask).squeeze(),axis=-1).reshape((53, 576, 768,1))[aaa[1]],cmap='gray')  # 第一个子图
axes[1, 0].imshow(np.argmax(np.array(test_mask).squeeze(),axis=-1).reshape((53, 576, 768,1))[aaa[2]],cmap='gray')  # 第一个子图
axes[1, 1].imshow(np.argmax(np.array(test_mask).squeeze(),axis=-1).reshape((53, 576, 768,1))[aaa[3]],cmap='gray')  # 第一个子图
plt.show()

# # In[ ]:


# import matplotlib.pyplot as plt
# # 创建一个包含四个子图的图像布局
# fig, axes = plt.subplots(2, 2, figsize=(7, 7))

# # 在每个子图中绘制内容
# axes[0, 0].imshow(np.argmax(np.array(test_pre_mask).squeeze(),axis=-1).reshape((53, 576, 768,1))[aaa[0]],cmap='gray')  # 第一个子图
# axes[0, 1].imshow(np.argmax(np.array(test_pre_mask).squeeze(),axis=-1).reshape((53, 576, 768,1))[aaa[1]],cmap='gray')  # 第一个子图
# axes[1, 0].imshow(np.argmax(np.array(test_pre_mask).squeeze(),axis=-1).reshape((53, 576, 768,1))[aaa[2]],cmap='gray')  # 第一个子图
# axes[1, 1].imshow(np.argmax(np.array(test_pre_mask).squeeze(),axis=-1).reshape((53, 576, 768,1))[aaa[3]],cmap='gray')  # 第一个子图


# # 分类指标

# In[ ]:


from sklearn.metrics import classification_report
test_predict_class_indices = np.argmax(np.array(test_pre_label).squeeze(), axis = 1)#找到预测类别是哪一个   哪个值最大就是哪一类
test_label = np.argmax(np.array(test_label).squeeze(), axis = 1)#找到预测类别是哪一个   哪个值最大就是哪一类
print(classification_report(test_label, test_predict_class_indices,digits=4))





# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_label, test_predict_class_indices)
print(cm)



# In[ ]:


np.array(test_pre_label).squeeze()


# In[ ]:


from sklearn.metrics import recall_score,precision_score,f1_score,roc_auc_score
roc = roc_auc_score(test_label, np.array(test_pre_label).squeeze()[:,1])
print('roc',roc)


# In[ ]:


# In[ ]:




# In[ ]:


test_loss = np.array(test_loss).mean()
print('test_loss',test_loss)



# In[ ]:


test_coef = np.array(test_coef).mean()
print('test_coef',test_coef)



# In[ ]:


test_iou = np.array(test_iou).mean()
print('test_iou',test_iou)



import matplotlib.pyplot as plt
# # 创建一个包含四个子图的图像布局
# fig, axes = plt.subplots(2, 2, figsize=(7, 7))

# # 在每个子图中绘制内容
# axes[0, 0].imshow(np.array(test_image).squeeze()[-1])  # 第一个子图
# axes[0, 1].imshow(np.array(test_image).squeeze()[-2])  # 第一个子图
# axes[1, 0].imshow(np.array(test_image).squeeze()[-10])  # 第一个子图
# axes[1, 1].imshow(np.array(test_image).squeeze()[28])  # 第一个子图


# # In[ ]:


# test_label[[-1,-2,-10]]


# # In[ ]:


# np.array(test_pre_label).squeeze()[[-1,-2,-10],:]


# # In[ ]:


# test_label[28],np.array(test_pre_label).squeeze()[28,:]


# # In[ ]:


# import time

# time.sleep(5*60)


# In[ ]:


# get_ipython().system('scancel 1729105')

