from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y
)

model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("Model Trained")

initial_types = [('input', FloatTensorType([None,4]))]

onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types)

with open('models/model.onnx','wb') as f:
    f.write(onnx_model.SerializeToString())
print("Model exported")