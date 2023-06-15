# Coding3FinalProject
## 概要
关于这次coding3的最终作业，想做一个深度学习项目，运用手势识别的数据集来改变摄像头图像的滤镜。我希望用这个模型来识别摄像头的图像手势，根据不同的手势展现不同的滤镜，改变摄像头图像。 我用visual studio code做了这个项目。
## 过程

## Machine learning algorithms:
I plan to use deep learning Convolutional Neural Networks (CNNs) for gesture recognition and a pre-trained neural network, such as MobileNet or VGG16, for feature extraction. For the filter application, I will use OpenCV, an image processing library.

## Existing code base: 
I will refer to some projects and tutorials from GitHub to speed up the development process. For example, I might use the code of this gesture recognition project as a reference:
[link of git]([https://github.com/这里写你GitHub项目的URL](https://github.com/opencv/opencv/tree/master/samples/dnn)）
In addition, I will also look at the official OpenCV documentation to learn how to apply filters:
https://docs.opencv.org/4.x/

## Dataset:
 I plan to use an image dataset containing multiple gestures to train my model. One available dataset is the "Hand Gesture Recognition Database" on Kaggle:
https://www.kaggle.com/datasets/gti-upm/leapgestrecog?resource=download

## Project Goal and Evaluation: 
My goal is to develop an application that can recognize gestures in real time and apply different style filters based on the recognized gestures. To evaluate the success of the project, I will conduct real-time tests to observe the accuracy of gesture recognition and the effectiveness and performance of filter application. The success criteria for the project are: a high level of accuracy in gesture recognition and the ability to apply filters to the camera image quickly and correctly based on the different gestures.

## 代码结构描述：

 1.Import Required Libraries: Essential libraries and modules like TensorFlow, NumPy, Keras' ImageDataGenerator, layers, models, etc., are imported.

 2.Check TensorFlow Version: Print the TensorFlow version to ensure it is compatible with the code.

 3.Dataset Path Setting: The script specifies the directory where the dataset is located.

 4.Data Augmentation: The ImageDataGenerator class is used to perform data augmentation for increasing the dataset size and improving generalization. This includes:

> · Scaling images
> · Rotating images
> · Shifting image width and height
> · Shearing images
> · Zooming in on images
> · Flipping images horizontally
> · Varying image brightness
> · Setting pixel fill mode

 5.Data Loading: Loads the dataset into training and validation datasets using flow_from_directory method, specifying image target size, batch size, class mode, subset (training or validation), and a random seed for reproducibility.

 6.Model Creation: Uses the EfficientNetB0 model as the base model. The weights of the model are initialized with pre-trained weights on the ImageNet dataset.

> · The base model is set to be trainable, and some of its layers are frozen for fine-tuning.
> · The final model is built by appending additional layers after the base model: GlobalAveragePooling2D, BatchNormalization, a Dense layer with ReLU activation and L2 regularization, a Dropout layer, and a final Dense layer with softmax activation for classification.

 7.Learning Rate Scheduling: Implements Cosine Decay for learning rate scheduling.

 8.Optimizer Setting: Uses the Adam optimizer with the specified learning rate schedule.

 9.Model Compilation: Compiles the model with the Adam optimizer, specifying the loss function as 'sparse_categorical_crossentropy', and monitoring the 'accuracy' metric.

 10.Training with Callbacks: Trains the model with the training data, validation data, and specifies the number of epochs. An early stopping callback is used to monitor the validation loss and stop the training if it doesn't improve for a certain number of epochs.

 11.Model Saving: Saves the trained model to disk for later use in predictions or further training.

> The script essentially performs image classification by training a deep neural network using the TensorFlow library. It uses an efficient pre-trained model (EfficientNetB0) and fine-tunes it for the specific task of hand gesture recognition. The script also includes data augmentation and learning rate scheduling to improve training efficiency and model generalization.
