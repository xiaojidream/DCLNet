import torch.nn as nn
import torch.nn.functional as F
import torch

class NIEnsembleNet(nn.Module):
    def __init__(self):
        super(NIEnsembleNet, self).__init__()
        self.classifier = nn.Linear(128+128, 4)

    def forward(self, x1, x2):
        # print(x1.shape, x2.shape)
        x2 = x2.view(x2.shape[0], -1)
        x = torch.cat([x1, x2], dim=1)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

