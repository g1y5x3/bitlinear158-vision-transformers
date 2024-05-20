import torch
from transformers import VitDetConfig, VitDetModel

config = VitDetConfig()
model = VitDetModel(config)

pixel_values = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    outputs = model(pixel_values)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)