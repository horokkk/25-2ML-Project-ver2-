import tensorflow as tf
from tensorflow.keras import layers, models

num_classes = 10   # UrbanSound8K classID 0~9

# (N, 40, 174) -> (N, 174, 40) 로 transpose
x_train_lstm = np.swapaxes(x_train, 1, 2)
x_val_lstm   = np.swapaxes(x_val,   1, 2)
x_test_lstm  = np.swapaxes(x_test,  1, 2)

print(x_train_lstm.shape)  # (7079, 174, 40)

lstm = models.Sequential([
    layers.Input(shape=(174, 40)),
    layers.LSTM(128),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

lstm.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_lstm = lstm.fit(
    x_train_lstm, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(x_val_lstm, y_val)
)

print("== LSTM Test 정확도 ==")
lstm.evaluate(x_test_lstm, y_test, verbose=2)
