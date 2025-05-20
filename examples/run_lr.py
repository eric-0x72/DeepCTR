import numpy as np
import tensorflow as tf
import tf2onnx

# 1. 构造简单数据
X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
y = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)

# 2. 构建线性回归模型：y = Wx + b
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model.compile(optimizer='adam', loss='mse')

# 3. 训练模型
model.fit(X, y, epochs=100, verbose=0)

# 4. 测试一下
print("Prediction:", model.predict(np.array([[4.0]], dtype=np.float32)))

# 5. 转换为 ONNX 格式
spec = (tf.TensorSpec((None, 1), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path="./onnx_files/lr_model.onnx")

print(">>> Saved lr_model.onnx")
