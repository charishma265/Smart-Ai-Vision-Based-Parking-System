import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("cnn_model.h5")

print("Model loaded successfully")

# Show model summary
model.summary()