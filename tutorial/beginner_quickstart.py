import tensorflow as tf
print("TensorFlow version: ", tf.__version__)

# Pixel values of images have range [0, 255]
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Scale values to the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Stack layers where each layer has one input tensor and one output tensor
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Model returns logits (log-odds scores) for each example
predictions = model(x_train[:1]).numpy()
print('Prediction logits:\n' + str(predictions))

# Softmax converts logits to probabilities for each class
print('Softmax\'ed logits:\n' + str(tf.nn.softmax(predictions).numpy()))

# Takes a vector of ground truth values and a vector of logits and returns
# a scalar loss for each example. This loss is equal to the negative log
# probablility of the true class--it is zero when the model is sure of the
# correct class.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print('Loss of one example:\n', loss_fn(y_train[:1], predictions).numpy())

# Configure and compile the model before training
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=5)

# Evaluate
model.evaluate(x_test, y_test, verbose=2)
