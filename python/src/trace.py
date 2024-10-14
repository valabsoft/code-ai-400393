import torch
from torchvision import models

# получение предобученной модели resnet34 используя PyTorch 
# Экземпляр вашей модели.
model = models.resnet34(pretrained=True)
# Переключите модель на eval model
model.eval()
# Пример входных данных, которые вы обычно вводите в метод forward() вашей модели.
var = torch.ones((1, 3, 224, 224))
# Используйте torch.jit.trace для создания модуля torch.jit.Script с помощью трассировки.
traced_script_module = torch.jit.trace(model, var)
traced_script_module.save("resnet34.pt")
