import torch
import torchvision
import coremltools as ct
from Moudle.YOLOv1 import YOLOv1Kai

cls = [
        "person","bird","cat","cow","dog","horse","sheep","aeroplane","bicycle","boat",
        "bus","car","motorbike","train","bottle","chair","diningtable","pottedplant","sofa","tvmonitor"
        ]
# Load a pre-trained version of MobileNetV2
torch_model = YOLOv1Kai(cls)
# Set the model in evaluation mode.
torch_model.load_state_dict(torch.load("checkpoints/231216/epoch99.pt"))
torch_model.eval()

# Trace the model with random data.
example_input = torch.rand(1, 3, 448, 448)
traced_model = torch.jit.trace(torch_model, example_input)
out = traced_model(example_input)
model = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)]
 )
model.save("checkpoints/mlmodel/YOLOv1Kai.mlpackage")