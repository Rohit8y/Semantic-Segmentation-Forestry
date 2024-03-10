import torch.nn as nn
import torchvision.models as models
from torchinfo import summary


class ResNet(nn.Module):

    def __init__(self, base_model, out_dim=128):
        super(ResNet, self).__init__()
        self.base_model = base_model
        self.resnet_dict = {"resnet18": models.resnet18(weights=None, num_classes=out_dim),
                            "resnet34": models.resnet34(weights=None, num_classes=out_dim),
                            "resnet50": models.resnet50(weights=None, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            print(
                "Invalid backbone architecture. Check the config file")
            print("Invalid mode name: ", model_name)
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = ResNet("resnet18", out_dim=512)
    txt = summary(model=model, input_size=(1, 3, 224, 224), depth=4)
