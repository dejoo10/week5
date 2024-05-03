# Activity 2:

There are 3 parts in this activity.

## Part 1/3: Introduction to Keras

1. **Import Libraries**:
```python
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
```

We begin by importing the necessary libraries. NumPy is imported to handle numerical computations, Matplotlib for visualization, Pandas for data manipulation, Keras for building neural network models, and train_test_split from scikit-learn for splitting the data into training and testing sets.

2. **Load Datasets**:
```python
cereal_data = pd.read_csv("cereal.csv")
concrete_data = pd.read_csv("concrete.csv")
```

We load the cereal and concrete datasets from CSV files. Ensure you have these files downloaded and placed in your working directory.

3. **Preprocess Datasets**:
```python
cereal_features = cereal_data[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']]
cereal_target = cereal_data['rating']

concrete_features = concrete_data.drop(columns=['CompressiveStrength'])
concrete_target = concrete_data['CompressiveStrength']
```
For simplicity, we preprocess each dataset by selecting a subset of features and treating the task as a regression problem. We extract features and target variables from each dataset.

4. **Define the Model**:
```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(cereal_X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

We define a Sequential model in Keras. A Sequential model allows you to create neural networks layer-by-layer. We add dense layers to the model, specifying the number of neurons and activation functions.

5. **Compile the Model**:
```python
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
```

After defining the model architecture, we compile it. Compiling the model involves specifying the optimizer, loss function, and metrics to monitor during training.

6. **Train the Model**:
```python
history_cereal = model.fit(cereal_X_train, cereal_y_train, epochs=10, batch_size=64, 
                           validation_data=(cereal_X_test, cereal_y_test), verbose=1)
```

We train the model on the cereal dataset using the fit method. We specify the training data, validation data, number of epochs, batch size, and verbosity level.

7. **Visualize Training History**:
```python
plt.plot(history_cereal.history['loss'], label='Cereal Train Loss')
plt.plot(history_cereal.history['val_loss'], label='Cereal Validation Loss')
plt.title('Training and Validation Loss for Cereal Dataset')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Finally, we plot the training and validation loss over epochs using Matplotlib. This allows us to visualize how the model's performance changes during training.

> Here's the [complete code](./lab2part1.py)

----
## Part 2/3: Early stopping

In Part 2, we'll modify the code to include early stopping. This modified prevents overfitting during model training. Early stopping allows the training process to stop if the validation loss does not improve for a certain number of epochs, thereby preventing overfitting.

**Part 2: Implementing Early Stopping**

1. **Import Libraries**:
```python
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
```

2. **Load and Preprocess Dataset**:
```python
cereal_data = pd.read_csv("cereal.csv")
cereal_features = cereal_data[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']]
cereal_target = cereal_data['rating']
cereal_X_train, cereal_X_test, cereal_y_train, cereal_y_test = train_test_split(cereal_features, cereal_target, test_size=0.2, random_state=42)
```

3. **Define the Model**:
```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(cereal_X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

4. **Compile the Model**:
```python
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
```

5. **Define Early Stopping Callback**:
```python
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
```

Step 5: **Define Early Stopping Callback**

In this step, we define an early stopping callback using Keras' `EarlyStopping` class. The purpose of this callback is to monitor the validation loss during training and stop the training process when the validation loss stops improving. 

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
```

Let's break down what each parameter in the `EarlyStopping` callback does:

- **monitor**: This parameter specifies the quantity to be monitored during training. In this case, we set it to `'val_loss'`, indicating that we want to monitor the validation loss.

- **patience**: Patience is the number of epochs with no improvement after which training will be stopped. In our code, we set it to `3`, meaning that training will stop if the validation loss does not decrease for 3 consecutive epochs.

- **verbose**: This parameter controls the verbosity mode. If set to `1`, it prints messages about the early stopping condition being triggered. If set to `0`, it operates silently. 

- **restore_best_weights**: When set to `True`, this parameter restores the model weights from the epoch with the lowest validation loss. This ensures that the model's performance is not adversely affected by training beyond the point of early stopping.

6. **Train the Model with Early Stopping**:
```python
history_cereal = model.fit(cereal_X_train, cereal_y_train, epochs=100, batch_size=64,
                            validation_data=(cereal_X_test, cereal_y_test),
                            callbacks=[early_stopping], verbose=1)
```

7. **Visualize Training History**:
```python
plt.plot(history_cereal.history['loss'], label='Cereal Train Loss')
plt.plot(history_cereal.history['val_loss'], label='Cereal Validation Loss')
plt.title('Training and Validation Loss for Cereal Dataset')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

> Here's the [complete code](./lab2part2.py)

> Does the early stopping callback help against overfitting?

The early stopping callback is an effective technique for combating overfitting. Here's how it helps:

1. **Prevents Overfitting**: Early stopping prevents overfitting by stopping the training process when the model's performance on the validation set starts to degrade. This ensures that the model does not continue to learn the idiosyncrasies of the training data at the expense of generalization to unseen data.

2. **Promotes Generalization**: By stopping training at an optimal point, early stopping promotes better generalization of the model to unseen data. It helps strike a balance between fitting the training data well and avoiding overfitting.

3. **Saves Computational Resources**: Early stopping saves computational resources by terminating training early when further iterations are unlikely to yield significant improvements. This is particularly useful when training deep neural networks that can be computationally intensive.

Overall, the early stopping callback is a valuable tool in the machine learning practitioner's arsenal for preventing overfitting and improving the generalization performance of neural network models.

----
## Part 3/3: Image Classification with Keras

In this part, we will walk through the process of building a convolutional neural network (CNN) for image classification using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. We will use the Keras deep learning library to create and train the CNN model.

**Step 1: Import Libraries**

First, let's import the necessary libraries for our project, including NumPy for numerical computations, Matplotlib for visualization, and the required modules from Keras for building and training our CNN.

```python
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
```

**Step 2: Load and Preprocess Data**

Next, we load the CIFAR-10 dataset using Keras' built-in `cifar10` module. We split the dataset into training and testing sets and normalize the pixel values to the range [0, 1]. We also convert the class labels to one-hot encoded vectors.

```python
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

**Step 3: Define the Model**

We define a Sequential model in Keras, which allows us to create a CNN layer by layer. We add convolutional layers with ReLU activation functions and max-pooling layers to downsample the feature maps. Finally, we flatten the output of the convolutional layers and add dense layers with ReLU activation functions. The output layer has a softmax activation function to output class probabilities.

```python
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

**Step 4: Compile the Model**

We compile the model by specifying the optimizer, loss function, and metrics to monitor during training. Here, we use the Adam optimizer, categorical cross-entropy loss function, and accuracy as the metric.

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

**Step 5: Train the Model**

Now, we train the model on the training data. We specify the number of epochs (iterations over the entire training dataset) and the batch size (number of samples per gradient update).

```python
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)
```

**Step 6: Evaluate the Model**

After training, we evaluate the model on the test data to assess its performance. We calculate the test loss and accuracy.

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
```

**Step 7: Visualize Training History**

Finally, we plot the training and validation accuracy over epochs to visualize the model's performance during training.

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

> Here's the [complete code](./lab2part3.py)