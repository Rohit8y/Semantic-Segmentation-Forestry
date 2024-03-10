import segmentation_models_pytorch as smp


def getDiceLoss():
    return smp.utils.losses.DiceLoss()


def getIoU():
    return [smp.utils.metrics.IoU(threshold=0.5)]
