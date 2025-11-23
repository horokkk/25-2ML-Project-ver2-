import tensorflow as tf
from tensorflow.keras import layers, models

num_classes = 10   # UrbanSound8K classID 0~9

# 채널 차원 추가 (H, W, C)
x_train_cnn = x_train[..., np.newaxis]  # (7079, 40, 174, 1)
x_val_cnn   = x_val[..., np.newaxis]
x_test_cnn  = x_test[..., np.newaxis]

print(x_train_cnn.shape)

cnn = models.Sequential([
    layers.Input(shape=(40, 174, 1)),
    
    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_cnn = cnn.fit(
    x_train_cnn, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(x_val_cnn, y_val)
)

print("== CNN Test 정확도 ==")
cnn.evaluate(x_test_cnn, y_test, verbose=2)
