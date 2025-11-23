import tensorflow as tf
from tensorflow.keras import layers, models

num_classes = 10   # UrbanSound8K classID 0~9

# 입력 평탄화
x_train_mlp = x_train.reshape(len(x_train), -1)
x_val_mlp   = x_val.reshape(len(x_val), -1)
x_test_mlp  = x_test.reshape(len(x_test), -1)

print(x_train_mlp.shape)  # (7079, 40*174)

mlp = models.Sequential([
    layers.Input(shape=(40*174,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

mlp.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_mlp = mlp.fit(
    x_train_mlp, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(x_val_mlp, y_val)
)

print("== MLP Test 정확도 ==")
mlp.evaluate(x_test_mlp, y_test, verbose=2)
