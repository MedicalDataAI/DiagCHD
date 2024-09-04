from keras.losses import CategoricalCrossentropy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from keras.applications.xception import Xception
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adagrad, Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.metrics import categorical_crossentropy

def dice_coeff(x, target, ignore_index = -100, epsilon=1e-6):
    d = 0.
    batch_size = tf.shape(x)[0]
    for i in range(batch_size):
        x_i = tf.reshape(x[i], [-1])
        t_i = tf.reshape(target[i], [-1])
        if ignore_index >= 0:
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
    dice = 0.
    for channel in range(tf.shape(x)[3]):
        dice += dice_coeff(x[:,:,:,channel], target[:,:,:,channel], ignore_index, epsilon)
    return dice / tf.cast(tf.shape(x)[3], dtype=tf.float32)

def dice_loss(x, target, multiclass = False, ignore_index = -100):
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)

def ce_dice_loss(y_true, y_pred):
    ce_loss = CategoricalCrossentropy()(y_true, y_pred)
    dice_coef = dice_loss(y_pred, y_true, multiclass=True)
    return ce_loss + dice_coef

def dice_coef(x, target, multiclass = True, ignore_index = -100):
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return fn(x, target, ignore_index=ignore_index)

# Ensure directories exist
dirs = ['./callbacks_EarlyStopping_a1', './callbacks_EarlyStopping_a2', 'log_a1', 'log_a2']
for dir in dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)

model = load_model(r'E:\zqx\az\CongenitalHeartDisease\callbacks_EarlyStopping.h5',
                   custom_objects={'ce_dice_loss':ce_dice_loss,'dice_coef':dice_coef})
model.save_weights('自定义分割模型权重.h5')

# 数据集的加载
height = 576  #图片的高度
width = 768  #图片的长度
channels = 3  #彩色图片
batch_size = 4*3  #每一次49个图片
test_size = 1
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

train_datagen_input_1 = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
train_generator_input_1 = train_datagen_input_1.flow_from_directory(
    train_dir_input_1,
    target_size=img_size_input_1,
    batch_size=batch_size,
    seed=SEED,
    shuffle=True,
    class_mode="categorical")
valid_datagen_input_1 = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
valid_generator_input_1 = valid_datagen_input_1.flow_from_directory(
    valid_dir_input_1,
    target_size=img_size_input_1,
    batch_size=batch_size,
    seed=SEED,
    shuffle=False,
    class_mode="categorical")

# mask创建
train_datagen_input_2 = keras.preprocessing.image.ImageDataGenerator()
train_generator_input_2 = train_datagen_input_2.flow_from_directory(
    train_dir_input_2,
    target_size=img_size_input_2,
    batch_size=batch_size,
    seed=SEED,
    shuffle=True,
    color_mode='grayscale',
    class_mode="categorical")
valid_datagen_input_2 = keras.preprocessing.image.ImageDataGenerator()
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
print(train_num_input_1, train_num_input_2, valid_num_input_1, valid_num_input_2)

from keras.losses import CategoricalCrossentropy
def cbam_block(cbam_feature, LayerName='', ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio, LayerName)
    cbam_feature = spatial_attention(cbam_feature, LayerName)
    return cbam_feature

def channel_attention(input_feature, ratio=8, LayerName=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros',
                             name=LayerName + 'Dense_1')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros',
                             name=LayerName + 'Dense_2')

    avg_pool = GlobalAveragePooling2D(name=LayerName + 'GlobalAveragePooling2D_1')(input_feature)
    avg_pool = Reshape((1, 1, channel), name=LayerName + 'Reshape_1')(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D(name=LayerName + 'GlobalAveragePooling2D_2')(input_feature)
    max_pool = Reshape((1, 1, channel), name=LayerName + 'Reshape_2')(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add(name=LayerName + 'Add')([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid', name=LayerName + 'Activation')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature, LayerName):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True), name=LayerName + 'Lambda_1')(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True), name=LayerName + 'Lambda_2')(cbam_feature)
    concat = Concatenate(axis=3, name=LayerName + 'Concatenate')([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same',
                          activation='sigmoid', kernel_initializer='he_normal', use_bias=False,
                          name=LayerName + 'Conv2D_1')(concat)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    return multiply([input_feature, cbam_feature])

def generate_data_generator(generator_input_1, generator_input_2):
    while True:
        x_data, label_x = generator_input_1.next()
        mask_data, label_mask = generator_input_2.next()
        assert np.array(label_x).all() == np.array(label_mask).all(), '数据集产出失败'
        yield np.array(x_data),{"Seg":tf.one_hot(np.array(tf.squeeze(mask_data,axis=-1)),depth=3),
                                "SegModel_Xception":label_x}

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3))
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x

def Unet_Xception_ResNetBlock(nClasses, input_height=224, input_width=224):
    backbone = Xception(input_shape=(input_height, input_width, 3), weights='imagenet', include_top=False)
    inputs = backbone.input
    conv4 = backbone.layers[121].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # Middle
    convm = Conv2D(16*32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, 16*32)
    convm = residual_block(convm, 16*32)
    convm = LeakyReLU(alpha=0.1,name='ClsLeakyReLU')(convm)

    # 8 -> 16
    deconv4 = Conv2DTranspose(16*16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = cbam_block(uconv4,'conv4')
    uconv4 = Conv2D(16*16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, 16 * 16)
    uconv4 = residual_block(uconv4, 16*16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(16*8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[31].output
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = cbam_block(uconv3,'conv3')
    uconv3 = Conv2D(16*8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, 16*8)
    uconv3 = residual_block(uconv3, 16*8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(16*4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[21].output
    conv2 = ZeroPadding2D(((1,0),(1,0)))(conv2)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = cbam_block(uconv2,'conv2')
    uconv2 = Conv2D(16*4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, 16*4)
    uconv2 = residual_block(uconv2, 16*4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(16*2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[11].output
    conv1 = ZeroPadding2D(((3,0),(3,0)))(conv1)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = cbam_block(uconv1,'conv1')
    uconv1 = Conv2D(16*2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, 16*2)
    uconv1 = residual_block(uconv1, 16*2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    # 128 -> 256
    uconv0 = Conv2DTranspose(16*1, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = Conv2D(16*1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, 16*1)
    uconv0 = residual_block(uconv0, 16*1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    out = Conv2D(nClasses, (1, 1), padding='same')(uconv0)
    out = BatchNormalization()(out)
    out = Activation('softmax',name="Seg")(out)

    model = Model(inputs=inputs, outputs=out)
    # model.load_weights('./自定义分割模型权重.h5')

    # 分类结果
    x = GlobalAveragePooling2D(name='SegModel_Xception_main_GlobalAveragePooling2D')(model.get_layer('ClsLeakyReLU').output)
    dp_1 = Dropout(0.6, name='SegModel_Xception_main_Dropout')(x)
    fc2_num_classes = Dense(2*500, kernel_initializer='he_normal', name='SegModel_Xception_main_Dense')(dp_1)
    dp_1 = Dropout(0.6, name='SegModel_Xception_main_Dropout1')(fc2_num_classes)
    fc2_num_classes = Dense(2*250, kernel_initializer='he_normal', name='SegModel_Xception_main_Dense1')(dp_1)
    dp_1 = Dropout(0.6, name='SegModel_Xception_main_Dropout2')(fc2_num_classes)
    fc2_num_classes = Dense(100, kernel_initializer='he_normal', name='SegModel_Xception_main_Dense2')(dp_1)
    dp_1 = Dropout(0.6, name='SegModel_Xception_main_Dropout3')(fc2_num_classes)
    fc2_num_classes = Dense(2, kernel_initializer='he_normal', name='SegModel_Xception_main_Dense_3')(dp_1)
    fc2_num_classes = Activation('softmax', name='SegModel_Xception')(fc2_num_classes)
    model = Model(inputs=model.input, outputs=[model.output,fc2_num_classes])
    return model

def setup_to_fine_tune_1(model):
    LayersNum = 0
    for layer in model.layers:
        if not layer.name.startswith('SegModel_Xception'):
            layer.trainable = False
            LayersNum += 1
    print('不可以训练的层有: ' + str(LayersNum) + "可以训练的层有: " + str(len(model.layers) - LayersNum))

    model.compile(optimizer=Adam(lr=0.001),
                  loss={"Seg":ce_dice_loss,
                                    "SegModel_Xception":categorical_crossentropy},
              loss_weights={'Seg': 0.1,
                      'SegModel_Xception': 1},
              metrics={"Seg":[dice_coef,tf.keras.metrics.OneHotMeanIoU(num_classes=3,name='miou')],
                                                         "SegModel_Xception":"accuracy"})

def setup_to_fine_tune_2(model):
    LayersNum = 0
    for layer in model.layers:
        layer.trainable = True
        LayersNum += 1
    print('不可以训练的层有: ' + str(LayersNum) + "可以训练的层有: " + str(len(model.layers) - LayersNum))

    model.compile(optimizer=Adam(lr=0.0001),loss={"Seg":ce_dice_loss,
                                    "SegModel_Xception":categorical_crossentropy},
              loss_weights={'Seg': 1,
                      'SegModel_Xception': 1},
              metrics={"Seg":[dice_coef,tf.keras.metrics.OneHotMeanIoU(num_classes=3,name='miou')],
                                                         "SegModel_Xception":"accuracy"})

model = Unet_Xception_ResNetBlock(3, height, width)
setup_to_fine_tune_1(model)

logdir = './callbacks_EarlyStopping_a1'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, "callbacks_EarlyStopping.h5")
log_dir = os.path.join('log_a1')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

#回调函数的使用-在训练中数据的保存
callbacks = [
    keras.callbacks.ModelCheckpoint(
        output_model_file,
        monitor='val_SegModel_Xception_accuracy',
        save_best_only=True,
        mode='max'),
    keras.callbacks.EarlyStopping(
        monitor='val_SegModel_Xception_accuracy', min_delta=1e-10, patience=53,mode='max'
    ),
    keras.callbacks.ReduceLROnPlateau(monitor='val_SegModel_Xception_accuracy',
                                        patience=10,
                                        mode='max',
                                        verbose=1,
                                        min_delta=1e-9),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
]

history = model.fit(generate_data_generator(train_generator_input_1,train_generator_input_2),validation_data=
        generate_data_generator(valid_generator_input_1,valid_generator_input_2),
        steps_per_epoch=train_num_input_1 // batch_size,
        epochs=epochs,
        validation_steps=valid_num_input_1 // test_size,
        callbacks=callbacks)

print('Saving model to disk\n')
model.save('model_Deep_ensemble_learning_1.h5')
print("history保存")
import pickle
with open('model_Deep_ensemble_learning_1.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# # 第二次训练
# model = load_model(r'callbacks_EarlyStopping.h5',custom_objects={'ce_dice_loss':ce_dice_loss,'dice_coef':dice_coef})
# setup_to_fine_tune_2(model)

# logdir = './callbacks_EarlyStopping_a2'
# if not os.path.exists(logdir):
#     os.mkdir(logdir)
# output_model_file = os.path.join(logdir, "callbacks_EarlyStopping.h5")
# log_dir = os.path.join('log_a2')
# if not os.path.exists(log_dir):
#     os.mkdir(log_dir)

# #回调函数的使用-在训练中数据的保存
# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         output_model_file,
#         monitor='val_SegModel_Xception_accuracy',
#         save_best_only=True,
#         mode='max'),
#     keras.callbacks.EarlyStopping(
#         monitor='val_SegModel_Xception_accuracy', min_delta=1e-10, patience=53,mode='max'
#     ),
#     keras.callbacks.ReduceLROnPlateau(monitor='val_SegModel_Xception_accuracy',
#                                         patience=10,
#                                         mode='max',
#                                         verbose=1,
#                                         min_delta=1e-9),
#     tf.keras.callbacks.TensorBoard(log_dir=log_dir),
# ]

# history = model.fit(generate_data_generator(train_generator_input_1,train_generator_input_2),validation_data=
#         generate_data_generator(valid_generator_input_1,valid_generator_input_2),
#         steps_per_epoch=train_num_input_1 // batch_size,
#         epochs=epochs,
#         validation_steps=valid_num_input_1 // test_size,
#         callbacks=callbacks)

# print('Saving model to disk\n')
# model.save('model_Deep_ensemble_learning_2.h5')
# print("history保存")
# with open('model_Deep_ensemble_learning_2.pickle', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)

# # 模型测试
# batch_size = 1
# train_generator_input_1 = train_datagen_input_1.flow_from_directory(
#     train_dir_input_1,
#     target_size=img_size_input_1,
#     batch_size=batch_size,
#     seed=SEED,
#     shuffle=True,
#     class_mode="categorical")
# valid_generator_input_1 = valid_datagen_input_1.flow_from_directory(
#     valid_dir_input_1,
#     target_size=img_size_input_1,
#     batch_size=batch_size,
#     seed=SEED,
#     shuffle=False,
#     class_mode="categorical")

# # mask创建
# train_generator_input_2 = train_datagen_input_2.flow_from_directory(
#     train_dir_input_2,
#     target_size=img_size_input_2,
#     batch_size=batch_size,
#     seed=SEED,
#     shuffle=True,
#     color_mode='grayscale',
#     class_mode="categorical")
# valid_generator_input_2 = valid_datagen_input_2.flow_from_directory(
#     valid_dir_input_2,
#     target_size=img_size_input_2,
#     batch_size=batch_size,
#     seed=SEED,
#     shuffle=False,
#     color_mode='grayscale',
#     class_mode="categorical")

# model = load_model('./callbacks_EarlyStopping_a2/callbacks_EarlyStopping.h5',
#                    custom_objects={'ce_dice_loss':ce_dice_loss,'dice_coef':dice_coef})

# test_label = []
# test_mask = []
# test_pre_label = []
# test_pre_mask = []
# test_image = []

# test_loss = []
# test_coef = []
# test_iou = []

# count = 0
# for image,label in generate_data_generator(valid_generator_input_1,valid_generator_input_2):
#     mask_label, y_label = label['Seg'], label['SegModel_Xception']
#     y_pre_mask, y_pre_label = model.predict(image)
    
#     test_loss.append(ce_dice_loss(mask_label, y_pre_mask))
#     test_coef.append(dice_coef(y_pre_mask, mask_label))
#     test_iou.append(tf.keras.metrics.OneHotMeanIoU(num_classes=3,name='miou')(mask_label, y_pre_mask))
    
#     test_label.append(y_label)
#     test_pre_label.append(y_pre_label)
    
#     test_image.append(image)
    
#     test_mask.append(mask_label)
#     test_pre_mask.append(y_pre_mask)
    
#     count += 1
#     if count == 53:
#         break

# # 百分位法的置信区间
# from sklearn import metrics
# import numpy as np
# def auc_boot(actual, predicted):
#     actual = np.array(actual)
#     predicted = np.array(predicted)
#     auc = metrics.accuracy_score(actual, predicted)

#     n_bootstraps = 10000
#     auc_scores = []
#     for i in range(n_bootstraps):
#         np.random.seed(i)
#         indices = np.random.choice(range(len(actual)), len(actual), replace=True)
#         auc_bootstrap = metrics.roc_auc_score(actual[indices], predicted[indices])
#         auc_scores.append(auc_bootstrap)
#     low, height = np.percentile(auc_scores, [2.5, 97.5])
#     return auc, low, height

# auc_boot(y_label, test_pre_label)

# np.array(test_loss).mean()
# np.array(test_coef).mean()
# np.array(test_iou).mean()

# import random
# random.seed = 666
# aaa = []
# aaa.append(random.randint(0, 52))
# aaa.append(random.randint(0, 52))
# aaa.append(random.randint(0, 52))
# aaa.append(random.randint(0, 52))

# fig, axes = plt.subplots(2, 2, figsize=(7, 7))
# axes[0, 0].imshow(np.array(test_image).squeeze()[aaa[0]])
# axes[0, 1].imshow(np.array(test_image).squeeze()[aaa[1]])
# axes[1, 0].imshow(np.array(test_image).squeeze()[aaa[2]])
# axes[1, 1].imshow(np.array(test_image).squeeze()[aaa[3]])

# fig, axes = plt.subplots(2, 2, figsize=(7, 7))
# axes[0, 0].imshow(np.argmax(np.array(test_mask).squeeze(), axis=-1).reshape((53, 576, 768, 1))[aaa[0]], cmap='gray')
# axes[0, 1].imshow(np.argmax(np.array(test_mask).squeeze(), axis=-1).reshape((53, 576, 768, 1))[aaa[1]], cmap='gray')
# axes[1, 0].imshow(np.argmax(np.array(test_mask).squeeze(), axis=-1).reshape((53, 576, 768, 1))[aaa[2]], cmap='gray')
# axes[1, 1].imshow(np.argmax(np.array(test_mask).squeeze(), axis=-1).reshape((53, 576, 768, 1))[aaa[3]], cmap='gray')

# fig, axes = plt.subplots(2, 2, figsize=(7, 7))
# axes[0, 0].imshow(np.argmax(np.array(test_pre_mask).squeeze(), axis=-1).reshape((53, 576, 768, 1))[aaa[0]], cmap='gray')
# axes[0, 1].imshow(np.argmax(np.array(test_pre_mask).squeeze(), axis=-1).reshape((53, 576, 768, 1))[aaa[1]], cmap='gray')
# axes[1, 0].imshow(np.argmax(np.array(test_pre_mask).squeeze(), axis=-1).reshape((53, 576, 768, 1))[aaa[2]], cmap='gray')
# axes[1, 1].imshow(np.argmax(np.array(test_pre_mask).squeeze(), axis=-1).reshape((53, 576, 768, 1))[aaa[3]], cmap='gray')

# # 分类指标
# from sklearn.metrics import classification_report
# test_predict_class_indices = np.argmax(np.array(test_pre_label).squeeze(), axis=1)
# test_label = np.argmax(np.array(test_label).squeeze(), axis=1)
# print(classification_report(test_label, test_predict_class_indices, digits=4))

# confusion_matrix(test_label, test_predict_class_indices)

# from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
# roc_auc_score(test_label, np.array(test_pre_label).squeeze()[:,1])

# fig, axes = plt.subplots(2, 2, figsize=(7, 7))
# axes[0, 0].imshow(np.array(test_image).squeeze()[-1])
# axes[0, 1].imshow(np.array(test_image).squeeze()[-2])
# axes[1, 0].imshow(np.array(test_image).squeeze()[-10])
# axes[1, 1].imshow(np.array(test_image).squeeze()[28])

# test_label[[-1, -2, -10]]
# np.array(test_pre_label).squeeze()[[-1, -2, -10], :]

# test_label[28], np.array(test_pre_label).squeeze()[28, :]

# import time
# time.sleep(5*60)


