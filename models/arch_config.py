import segmentation_models_pytorch as smp


class DeepLabModel():
    def __init__(self, arch="resnet18", pretrained="imagenet", classes=2, activation="sigmoid"):
        self.arch = arch
        self.pretrained = pretrained
        # create segmentation model with pretrained encoder
        self.model = smp.DeepLabV3Plus(
            encoder_name=arch,
            encoder_weights=pretrained,
            classes=classes,
            activation=activation,
        )

    def get_model(self):
        return self.model

    def get_preprocessing(self):
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.arch, self.pretrained)
        return preprocessing_fn


if __name__ == "__main__":
    deepObj = DeepLabModel()
    print(deepObj.get_model())
