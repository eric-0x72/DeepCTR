import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat


# Custom data reader for calibration
class DummyDataReader(CalibrationDataReader):
    def __init__(self, calibration_data):
        self.data = calibration_data
        self.enum_data = iter(calibration_data)

    def get_next(self):
        return next(self.enum_data, None)


# Example calibration data
# calibration_data = [{"input": np.array([[0.8], [0.2]], dtype=np.float32)}]  # Replace with your real input
calibration_data = [
    {
        "user:0": np.array([[0]], dtype=np.int32),
        "gender:0": np.array([[0]], dtype=np.int32),
        "item_id:0": np.array([[1]], dtype=np.int32),
        "cate_id:0": np.array([[1]], dtype=np.int32),
        "pay_score:0": np.array([[0.1]], dtype=np.float32),
        "hist_item_id:0": np.array([[1, 2, 3, 0]], dtype=np.int32),
        "hist_cate_id:0": np.array([[1, 2, 2, 0]], dtype=np.int32),
        "seq_length:0": np.array([[3]], dtype=np.int32),
    },
]

reader = DummyDataReader(calibration_data)

quantize_static(
    model_input="din.onnx",
    model_output="din_int8.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QOperator,
)
