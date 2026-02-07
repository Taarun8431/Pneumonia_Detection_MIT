import torch
import torchxrayvision as xrv
from src.utils import get_model

model = get_model()
print(model.features)
