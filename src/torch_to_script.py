import torch
import torchvision

# An instance of your model.
model = torch.load("models/cifar.ckpt", map_location=torch.device('cpu'))

model.load_state_dict(model["state_dict"])
# Switch the model to eval model
# model.eval()

# An example input you would normally provide to your model's forward() method.
# example = torch.rand(1, 3, 224, 224)

# Use torch.jit.script to generate a torch.jit.ScriptModule via scripting.
traced_script_module = torch.jit.script(model)

# Save the TorchScript model
traced_script_module.save("traced_resnet_model.script.pt")