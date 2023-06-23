from torch import nn
from torchvision.models import shufflenet_v2_x0_5



def get_model(device):
    model = shufflenet_v2_x0_5(True)
    model.fc = nn.Linear(1024, 1)
    return model.to(device)
