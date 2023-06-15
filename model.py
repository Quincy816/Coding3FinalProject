import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2

print("TensorFlow version:", tf.__version__)

dataset_path = 'D:\coding3final\leapGestRecog'

# 扩展数据增强
data_generator = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode="nearest"
)

train_dataset = data_generator.flow_from_directory(
    dataset_path,
    target_size=(96, 96),
    batch_size=32,
    class_mode='sparse',
    subset='training',
    seed=123
)

val_dataset = data_generator.flow_from_directory(
    dataset_path,
    target_size=(96, 96),
    batch_size=32,
    class_mode='sparse',
    subset='validation',
    seed=123
)

# 使用EfficientNetB0替换MobileNetV2
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model.trainable = True

# 选择解冻的层数
fine_tune_at = len(base_model.layers) // 2
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 构建模型
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(11, activation='softmax')
])

# 使用余弦退火学习率调度
initial_learning_rate = 0.01
decay_steps = int(train_dataset.samples / 32) * 10  # Assuming 10 epochs
cosine_decay = CosineDecay(initial_learning_rate, decay_steps)

# 使用Adam优化器
optimizer = Adam(cosine_decay)

# 编译模型
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 使用学习率调整和早停
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# 训练模型
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[early_stopping]
)

# 保存模型
model.save('hand_gesture_model_optimized')
