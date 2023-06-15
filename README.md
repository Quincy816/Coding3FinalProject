# Coding3FinalProject

## YouTube Link
https://youtu.be/03QAWyLoR48

## Outline
About this coding3 final assignment, I want to do a deep learning project to use the dataset of gesture recognition to change the filter of the camera image. I want to use this model to recognize camera image gestures and show different filters to change the camera image according to different gestures. I have made this project in visual studio code.

## Process
As soon as I started training, I found that the accuracy of my training results was too low, only about 50, and the dataset was more accurate on the training set but very low on the validation set. Then I thought that maybe the model was overfitted during training, so I used some data augmentation and normalization, added EarlyStopping callbacks to avoid overfitting, and added a learning rate scheduler. Then it was much better.
Then I set the gesture to #10 as a green filter and #02 as a red filter to prove that my verification was accurate. Then I added lines depicting close-ups of the hand in the filter.py file to make the display screen more explicit.

## Machine learning algorithms:

1. **EfficientNetB0**: This is a neural network architecture for image classification tasks. It's a part of the EfficientNet family which is known for efficient scaling of neural networks. The line `base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(96, 96, 3))` initializes the architecture with pre-trained weights on ImageNet.

2. **Data Augmentation**: Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data. Data augmentation techniques such as rotations, scaling, etc. are applied to the training data. This is done through the `ImageDataGenerator` class.

3. **Global Average Pooling**: Global Average Pooling is a layer that is used frequently in deep learning models, especially for image classification. It replaces the traditional fully connected layer in CNN by taking the average of all values in the feature map. In your code, it's used as `layers.GlobalAveragePooling2D()`.

4. **Batch Normalization**: This is used to normalize the activations of the neurons in a network between layers. It can stabilize and accelerate the training of deep neural networks. In your code, it's used as `layers.BatchNormalization()`.

5. **Dropout**: It's a regularization technique where during training, randomly selected neurons are ignored or "dropped-out". This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass. This is done using `layers.Dropout(0.5)`.

6. **L2 Regularization**: This is a penalty added to the loss function that scales with the square of the magnitude of weights. It's used to prevent overfitting in the neural network. In your code, it's used as `kernel_regularizer=l2(0.01)`.

7. **Softmax Activation Function**: Used in the output layer of the neural network for multiclass classification problems. It turns logits into probabilities that sum to one. It is used in `layers.Dense(11, activation='softmax')`.

8. **Adam Optimizer with Cosine Decay**: Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent to update network weights iteratively based on the training data. Cosine decay is used to adjust the learning rate during training for better convergence. It's set using `optimizer = Adam(cosine_decay)`.

9. **Sparse Categorical Crossentropy Loss Function**: It is used as a loss function for training classification problems with multiple classes. It is used when the target is in the form of integers.

10. **Early Stopping**: This is used to stop the training when a monitored quantity has stopped improving, which helps in preventing overfitting. This is done using `EarlyStopping`.

This code is for training a deep learning model (specifically EfficientNetB0) for image classification with data augmentation, regularization, and custom learning rate schedule.

## Existing code base: 
I will refer to some projects and tutorials from GitHub to speed up the development process. For example, I might use the code of this gesture recognition project as a reference:
> https://github.com/opencv/opencv/tree/master/samples/dnn

In addition, I will also look at the official OpenCV documentation to learn how to apply filters:
> https://docs.opencv.org/4.x/

## Dataset:
 I plan to use an image dataset containing multiple gestures to train my model. One available dataset is the "Hand Gesture Recognition Database" on Kaggle:
> https://www.kaggle.com/datasets/gti-upm/leapgestrecog?resource=download

## Project Goal and Evaluation: 
My goal is to develop an application that can recognize gestures in real time and apply different style filters based on the recognized gestures. To evaluate the success of the project, I will conduct real-time tests to observe the accuracy of gesture recognition and the effectiveness and performance of filter application. The success criteria for the project are: a high level of accuracy in gesture recognition and the ability to apply filters to the camera image quickly and correctly based on the different gestures.

## Code structure description:

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
