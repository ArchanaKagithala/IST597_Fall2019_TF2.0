

# ============================================================
# CS 599: Foundations of Deep Learning
# Assignment #00001 - Problem 1: Linear Regression
# Author: Archana K, Fall 2024 student
# ============================================================

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------
# 1️⃣  Reproducibility (unique seed from name -> Archana)
# ------------------------------------------------------------
seed = 1759793959
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ------------------------------------------------------------
# 2️⃣  Generate synthetic data: y = 3x + 2 + noise
# ------------------------------------------------------------
def make_noise(n, kind='gaussian', scale=1.0):
    if kind == 'gaussian':
        return np.random.normal(0, scale, size=n)
    elif kind == 'uniform':
        return np.random.uniform(-scale, scale, size=n)
    elif kind == 'laplace':
        return np.random.laplace(0, scale, size=n)
    else:
        raise ValueError("Unknown noise type")

N = 10000
true_W, true_b = 3.0, 2.0
x = np.random.uniform(-10, 10, N).astype(np.float32)
y = (true_W * x + true_b + make_noise(N, 'gaussian', scale=1.0)).astype(np.float32)

# ------------------------------------------------------------
# 3️⃣  Dataset pipeline
# ------------------------------------------------------------
batch_size = 128
dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000, seed=seed).batch(batch_size)

# ------------------------------------------------------------
# 4️⃣  Define model parameters and initialization
# ------------------------------------------------------------
def init_weights(init_type='normal'):
    if init_type == 'zeros':
        return tf.Variable(0.0), tf.Variable(0.0)
    elif init_type == 'normal':
        return tf.Variable(tf.random.normal([], stddev=0.1, seed=seed)), tf.Variable(0.0)
    elif init_type == 'large':
        return tf.Variable(tf.random.normal([], stddev=5.0, seed=seed)), tf.Variable(5.0)
    else:
        raise ValueError("Unknown init type")

W, b = init_weights('normal')

# ------------------------------------------------------------
# 5️⃣  Define loss functions
# ------------------------------------------------------------
def mse_loss(y, yhat): return tf.reduce_mean(tf.square(y - yhat))
def mae_loss(y, yhat): return tf.reduce_mean(tf.abs(y - yhat))
def huber_loss(y, yhat, delta=1.0):
    r = y - yhat
    return tf.reduce_mean(tf.where(tf.abs(r) <= delta, 0.5 * tf.square(r), delta * (tf.abs(r) - 0.5*delta)))
def hybrid_loss(y, yhat, alpha=0.5):
    return alpha * mae_loss(y, yhat) + (1 - alpha) * mse_loss(y, yhat)

loss_fn = hybrid_loss   # choose: mse_loss, mae_loss, huber_loss, hybrid_loss

# ------------------------------------------------------------
# 6️⃣  Optimizer and learning rate scheduling
# ------------------------------------------------------------
lr_var = tf.Variable(0.1, trainable=False)
optimizer = tf.optimizers.SGD(learning_rate=lr_var)
patience, best_loss, epochs_since_improve = 5, 1e9, 0

# ------------------------------------------------------------
# 7️⃣  Training loop with noise and patience scheduling
# ------------------------------------------------------------
EPOCHS = 50
train_losses, lr_log = [], []

print("Starting training...")
for epoch in range(EPOCHS):
    t0 = time.perf_counter()
    epoch_loss = 0.0
    for xb, yb in dataset:
        with tf.GradientTape() as tape:
            yhat = W * xb + b
            loss = loss_fn(yb, yhat)
        grads = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(grads, [W, b]))
        epoch_loss += loss.numpy()

    epoch_loss /= len(dataset)
    train_losses.append(epoch_loss)
    lr_log.append(lr_var.numpy())

    # Learning rate scheduling (patience)
    if epoch_loss < best_loss - 1e-4:
        best_loss = epoch_loss
        epochs_since_improve = 0
    else:
        epochs_since_improve += 1
        if epochs_since_improve >= patience:
            lr_var.assign(lr_var * 0.5)
            epochs_since_improve = 0

    # Add random noise to weights/learning rate occasionally
    if epoch % 10 == 0:
        W.assign_add(tf.random.normal([], stddev=0.001))
        lr_var.assign(lr_var * (1 + np.random.uniform(-0.05, 0.05)))

    t1 = time.perf_counter()
    print(f"Epoch {epoch+1:02d}: loss={epoch_loss:.6f}, W={W.numpy():.3f}, b={b.numpy():.3f}, "
          f"lr={lr_var.numpy():.4f}, time={t1-t0:.3f}s")

# ------------------------------------------------------------
# 8️⃣  Results and plotting
# ------------------------------------------------------------
print("\nTraining complete.")
print(f"Final parameters: W={W.numpy():.4f}, b={b.numpy():.4f}")

# Plot loss curve
plt.figure(figsize=(6,4))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.tight_layout()
plt.savefig("linreg_loss.png")
plt.show()

# Plot fitted line vs true data
plt.figure(figsize=(6,4))
plt.scatter(x[:200], y[:200], s=10, label="Data", alpha=0.6)
plt.plot(x, W.numpy()*x + b.numpy(), color='r', label="Model Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Fitted Line")
plt.tight_layout()
plt.savefig("linreg_fit.png")
plt.show()
