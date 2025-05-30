# -*- coding: utf-8 -*-
"""TSMAE algorithm by Ngo Khac Bao Long""

Original file is located at
    https://colab.research.google.com/drive/1Onx3Ujh6HW7U9isLrKXriG3cCEHCKk6C
"""

import pandas as pd
import numpy as np

# Load the complete ECG5000 dataset
data = pd.read_csv("ecg.csv", header=None)

# Separate features and labels
X = data.iloc[:, :-1]  # All columns except the last one
labels = data.iloc[:, -1]  # The last column is the label

# Separate normal and abnormal samples
normal_samples = data[labels == 0]
abnormal_samples = data[labels == 1]

# Randomly select 293 normal and 28 abnormal samples
selected_normal_samples = normal_samples.sample(n=293, random_state=42)
selected_abnormal_samples = abnormal_samples.sample(n=28, random_state=42)

# Combine the selected samples
selected_data = pd.concat([selected_normal_samples, selected_abnormal_samples])

# Shuffle the selected data to mix normal and abnormal samples
selected_data = selected_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels again after sampling
X_selected = selected_data.iloc[:, :-1].values
labels_selected = selected_data.iloc[:, -1].values

# Initialize an array to store normalized data
X_normalized = np.zeros_like(X_selected)

# Apply Min-Max scaling for each column
for col in range(X_selected.shape[1]):
    col_min = X_selected[:, col].min()
    col_max = X_selected[:, col].max()

    # Avoid division by zero if col_min equals col_max
    if col_max - col_min != 0:
        X_normalized[:, col] = (X_selected[:, col] - col_min) / (col_max - col_min)
    else:
        X_normalized[:, col] = 0  # or any constant, as all values in this column are the same

# Convert normalized data back to DataFrame and add the labels
normalized_data = pd.DataFrame(X_normalized)
normalized_data['label'] = labels_selected

# Save the preprocessed data if needed
normalized_data.to_csv("ECG5000_normalized_sampled.csv", index=False)

import tensorflow as tf
from tensorflow.keras import layers, Model

# Define the LSTM Encoder model
class LSTMEncoder(Model):
    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.lstm = layers.LSTM(hidden_size, activation='sigmoid', return_state=True)

    def call(self, x):
        # Forward pass through LSTM; only keep the final hidden state (h) as the latent representation
        _, h, _ = self.lstm(x)
        z = h  # Latent representation
        return z

# Parameters
T = 140 # Number of time steps per sample
hidden_size = 10  # Size of the hidden layer (latent representation)
batch_size = 20 # Number of samples in each batch
num_features = 1 # Number of features per time step (single acquisition per action)

# Instantiate the model
encoder = LSTMEncoder(input_size=num_features,hidden_size=hidden_size)

X_normalized = tf.expand_dims(X_normalized, -1)  # Shape becomes (321, 140, 1)
# Get latent representation
z = encoder(X_normalized)
print("Latent representation z shape:", z.shape)  # Expected shape: (321, hidden_size)

import tensorflow as tf

# Parameters
E = 10  # Dimension of latent representation
N = 20  # Number of memory items
lambda_threshold = 1 / N  # Sparsification threshold, lambda >= 1/N
epsilon = 1e-10  # Small value to avoid division by zero in normalization

# Step 1: Initialize M with Xavier initialization
initializer = tf.keras.initializers.GlorotUniform()
M = tf.Variable(initializer(shape=(N, E)), trainable=True, dtype=tf.float32)


# Step 3: Calculate addressing vector q
similarity_scores = tf.matmul(z, M, transpose_b=True)  # Inner product with each memory item
q = tf.nn.softmax(similarity_scores, axis=1)  # Apply softmax to get initial addressing vector

# Step 4: Rectify q based on the lambda_threshold
q_rectified = (tf.maximum(q - lambda_threshold, 0) * q) / abs(q - lambda_threshold)

# Step 5: Normalize using L1 normalization with epsilon to stabilize
q_l1_norm = tf.reduce_sum(tf.abs(q_rectified), axis=1, keepdims=True)
q_normalized = q_rectified / tf.maximum(q_l1_norm, epsilon)

# Final addressing vector
print("Final addressing vector shape:", q_normalized.shape)
print("Sum of each row in q after normalization:", tf.reduce_sum(q_normalized, axis=1))

# Assuming q_normalized is the final addressing vector obtained from the previous steps
# and M is the memory matrix

# Calculate the reconstructed latent representation z_hat
z_hat = tf.matmul(q_normalized, M)

print("Reconstructed latent representation (z_hat) shape:", z_hat.shape)

import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Define the LSTM Decoder model
class LSTMDecoder(tf.keras.Model):
    def __init__(self, sequence_length, latent_dim, dropout_rate=0.2):
        super(LSTMDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # Define the layers in the LSTM decoder
        self.lstm_decoder = Sequential([
            layers.RepeatVector(sequence_length),                   # Repeat z_hat for each time step
            layers.LSTM(sequence_length, return_sequences=True),    # First LSTM layer
            layers.Dropout(dropout_rate),                           # Dropout layer
            layers.LSTM(sequence_length, return_sequences=True),    # Second LSTM layer
            layers.Dropout(dropout_rate),                           # Dropout layer
            layers.TimeDistributed(layers.Dense(1))                # Output layer for each time step
        ])

    def call(self, z_hat):
        # Pass the latent representation through the LSTM decoder layers
        x_hat = self.lstm_decoder(z_hat)
        # Reshape the output to match the shape of X_normalized
        # The -1 ensures batch size is handled automatically
        # sequence_length and 1 provide the correct dimensions
        #x_hat = tf.reshape(x_hat, [-1, sequence_length, 1])

        return x_hat # Remove tf.squeeze to preserve all dimensions


# Instantiate the LSTM decoder
sequence_length = 140  # Length of the original sequence
latent_dim = 10        # Dimensionality of the latent representation z_hat
decoder = LSTMDecoder(sequence_length=sequence_length, latent_dim=latent_dim, dropout_rate=0.2)

# Example input to the decoder

x_hat = decoder(z_hat)  # Get the reconstructed sequence

print("Reconstructed sample (x_hat) shape:", x_hat.shape)

# Fixed sparsity coefficient
eta = 0.01 # Theo bài báo TSMAE

import tensorflow as tf
from tensorflow.keras import layers, Model

# ... (Your existing code for LSTMEncoder, LSTMDecoder, and memory module) ...

class TSMAE(Model):
    def __init__(self, input_size, hidden_size, sequence_length, latent_dim, dropout_rate=0.2, N=20, E=10, lambda_threshold=0.05, epsilon=1e-10):
        super(TSMAE, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size)
        self.decoder = LSTMDecoder(sequence_length, latent_dim, dropout_rate)

        # Memory module parameters
        self.N = N  # Number of memory items
        self.E = E  # Dimension of latent representation
        self.lambda_threshold = lambda_threshold  # Sparsification threshold
        self.epsilon = epsilon  # Small value to avoid division by zero

        # Initialize M with Xavier initialization
        initializer = tf.keras.initializers.GlorotUniform()
        self.M = tf.Variable(initializer(shape=(self.N, self.E)), trainable=True, dtype=tf.float32)

    def call(self, inputs):
        # Encoder
        z = self.encoder(inputs)

        # Memory Module
        similarity_scores = tf.matmul(z, self.M, transpose_b=True)
        q = tf.nn.softmax(similarity_scores, axis=1)
        q_rectified = (tf.maximum(q - self.lambda_threshold, 0) * q) / abs(q - self.lambda_threshold)
        q_l1_norm = tf.reduce_sum(tf.abs(q_rectified), axis=1, keepdims=True)
        q_normalized = q_rectified / tf.maximum(q_l1_norm, self.epsilon)

        # Decoder
        x_hat = self.decoder(tf.matmul(q_normalized, self.M))

        return x_hat  # Trả về list


# Fixed sparsity coefficient
eta = 0.01  # Theo bài báo TSMAE

def custom_loss(y_true, y_pred):
    """
    Custom loss function combining reconstruction loss and sparsity loss.

    Args:
        y_true: The original input data (ground truth).
        y_pred: A tuple containing the reconstructed data (x_hat) and q_normalized.

    Returns:
        The total loss: reconstruction loss + eta * sparsity loss
    """

    # Truy cập các giá trị từ list
    x_hat = y_pred[0]




    # Reconstruction Loss (Mean Squared Error)
    reconstruction_loss = tf.reduce_mean(tf.square(y_true - x_hat)) / 2.0



    # Sparsity Loss (log sparsity penalty)
    q_normalized = q_rectified / tf.maximum(q_l1_norm, epsilon)
    sparsity_loss = tf.reduce_sum(-tf.math.log(1 + tf.square(q_normalized)))

    # Tổng loss
    total_loss = reconstruction_loss + eta * sparsity_loss

    return total_loss

# Khởi tạo model, providing the necessary arguments
input_size = 1  # Number of features (from your previous code)
hidden_size = 10 # Size of the hidden layer (from your previous code)
sequence_length = 140 # Length of the sequence (from your previous code)
latent_dim = 10 # Dimension of the latent representation (from your previous code)

model = TSMAE(input_size=input_size,
              hidden_size=hidden_size,
              sequence_length=sequence_length,
              latent_dim=latent_dim)

# Compile model
# Pass your custom loss function 'custom_loss' to the 'loss' argument
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=custom_loss,  # Use your custom loss function
              metrics=['mse'])  # Sử dụng metric phù hợp

# Huấn luyện model
model.fit(X_normalized, X_normalized, epochs=50, batch_size=20)

# Assuming 'model' is your trained TSMAE model and 'X_normalized' is your input data

# 1. Get the latent representation (z) for your actual data:
z = model.encoder(X_normalized)

# 2. Access the memory module parameters from your model:
M = model.M
lambda_threshold = model.lambda_threshold
epsilon = model.epsilon

# 3. Calculate q_normalized using the latent representation and memory module parameters:
similarity_scores = tf.matmul(z, M, transpose_b=True)
q = tf.nn.softmax(similarity_scores, axis=1)
q_rectified = (tf.maximum(q - lambda_threshold, 0) * q) / abs(q - lambda_threshold)
q_l1_norm = tf.reduce_sum(tf.abs(q_rectified), axis=1, keepdims=True)
q_normalized = q_rectified / tf.maximum(q_l1_norm, epsilon)

# Now 'q_normalized' contains the correct addressing vectors for your data 'X_normalized'
print("q_normalized values:")
print(q_normalized[:5])

# Assuming q_normalized is a TensorFlow tensor
sparsity_level = tf.reduce_mean(tf.cast(tf.equal(q_normalized, 0), tf.float32))

print("Sparsity Level:", sparsity_level.numpy())

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

# ... (Your existing code for LSTMEncoder, LSTMDecoder, TSMAE, custom_loss, etc.) ...

# Assuming 'model' is your trained TSMAE model and 'X_normalized' is your input data

# 1. Get the latent representation (z) for your actual data:
z = model.encoder(X_normalized)

# 2. Access the memory module parameters from your model:
M = model.M
lambda_threshold = model.lambda_threshold
epsilon = model.epsilon

# 3. Calculate q_normalized using the latent representation and memory module parameters:
similarity_scores = tf.matmul(z, M, transpose_b=True)
q = tf.nn.softmax(similarity_scores, axis=1)
q_rectified = (tf.maximum(q - lambda_threshold, 0) * q) / abs(q - lambda_threshold)
q_l1_norm = tf.reduce_sum(tf.abs(q_rectified), axis=1, keepdims=True)
q_normalized = q_rectified / tf.maximum(q_l1_norm, epsilon)

# 4. Randomly select 14 rows from q_normalized
num_test_samples = 14
random_indices = np.random.choice(q_normalized.shape[0], num_test_samples, replace=False)

# Convert random_indices to a list of integers for tf.gather
random_indices_list = random_indices.tolist()

# Select the rows from q_normalized using tf.gather
q_normalized_sampled = tf.gather(q_normalized, random_indices_list)

# 5. Create the heatmap for the 14 sampled rows with smaller size and blue color
plt.figure(figsize=(6, 4))  # Reduced figure size to 6x4 inches
vmax_value = q_normalized.numpy().max()  # Dynamically adapt vmax to 0.13
sns.heatmap(q_normalized_sampled.numpy(), cmap="Blues", cbar=True, vmin=0, vmax=vmax_value)  # Changed cmap to "Blues"
plt.title("Addressing Vector (q_normalized) Heatmap", fontsize=12)  # Reduced font size
plt.xlabel("Memory Items", fontsize=10)  # Reduced font size
plt.ylabel("Data Samples (14)", fontsize=10)  # Reduced font size
plt.tight_layout()
plt.show()

# Dự đoán trên dữ liệu huấn luyện
X_reconstructed = model.predict(X_normalized)

# Tính toán lỗi tái tạo
reconstruction_error = np.mean(np.square(X_normalized - X_reconstructed), axis=1)

# In ra lỗi tái tạo trung bình
print("Average Reconstruction Error:", np.mean(reconstruction_error))

import matplotlib.pyplot as plt

plt.hist(reconstruction_error, bins=50, color='blue', alpha=0.7)
plt.title("Distribution of Reconstruction Errors")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()

anomaly_ratio = 8.75 / 100
threshold = np.percentile(reconstruction_error, 100 - anomaly_ratio * 100)
print(f"Threshold based on anomaly ratio: {threshold}")

anomalies = reconstruction_error > threshold  # True nếu lỗi tái tạo lớn hơn ngưỡng
print(f"Number of anomalies detected: {np.sum(anomalies)}")
print(f"Number of normal samples: {len(reconstruction_error) - np.sum(anomalies)}")

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

y_pred = (reconstruction_error > threshold).astype(int)  # 1: anomaly, 0: normal
y_true = (labels_selected != 0).astype(int)  # 1: anomaly, 0: normal (giả sử nhãn của bạn)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, reconstruction_error)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, AUC: {auc:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))