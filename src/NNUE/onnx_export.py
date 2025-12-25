import torch as pt
from NNUE.model import NNUE


# Define the paths for the input PyTorch model and the output ONNX model.
pytorch_model_path = "NNUE/models/nnue.pth"
onnx_model_path = "NNUE/models/nnue.onnx"
export_full = False

# Instantiate the model with the default architecture.
# This must match the architecture of the saved .pth file.
model = NNUE([512])

# Load the trained weights into the model.
try:
    model.load(pytorch_model_path)
    print(f"Successfully loaded model from {pytorch_model_path}")
except Exception as e:
    print(f"Failed to load model. Ensure the file exists and the architecture matches. Error: {e}")
    exit()

# Set the model to evaluation mode. This is crucial for consistent inference.
model.eval()

if export_full:
    # Create a dummy input tensor with the correct shape (batch_size, input_features).
    dummy_input = pt.randn(1, 768)
else:
    model = model.layers
    dummy_input = pt.randn(1, 512)

# Export the model to ONNX format.
pt.onnx.export(model, dummy_input, onnx_model_path, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, opset_version=11)
print(f"Model successfully exported to {onnx_model_path}")