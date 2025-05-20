from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import onnx
import numpy as np


# Custom data reader for calibration
class DummyDataReader(CalibrationDataReader):
    def __init__(self, calibration_data):
        self.data = calibration_data
        self.enum_data = iter(calibration_data)

    def get_next(self):
        return next(self.enum_data, None)


# Example calibration data
calibration_data = [{"input": np.array([[0.8], [0.2]], dtype=np.float32)}]  # Replace with your real input

reader = DummyDataReader(calibration_data)

quantize_static(
    model_input="lr_model.onnx",
    model_output="lr_model_int8.onnx",
    calibration_data_reader=reader,
    quant_format=QuantFormat.QOperator,
)
