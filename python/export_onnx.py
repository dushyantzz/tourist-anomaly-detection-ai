import joblib
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import pandas as pd

def export_joblib_to_onnx(model_path, export_path, shape):
    model = joblib.load(model_path)
    initial_type = [('float_input', FloatTensorType([None, shape]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(export_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

if __name__ == "__main__":
    # adjust shape (num features) if needed
    shape = pd.read_csv('./data/X_train.csv').shape[1]
    export_joblib_to_onnx('./deployment/model_lgbm.pkl', './deployment/model_lgbm.onnx', shape)
    export_joblib_to_onnx('./deployment/model_xgb.pkl', './deployment/model_xgb.onnx', shape)
    print('Exported LightGBM and XGBoost to ONNX!')
