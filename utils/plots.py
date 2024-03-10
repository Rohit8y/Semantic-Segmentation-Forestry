import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


def generate_plots(args, train_logs_list, test_logs_list):
    # Arrays to store the loss and iou score
    trainDice = []
    testDice = []
    trainIou = []
    testIou = []

    for log in train_logs_list:
        trainDice.append(log['dice_loss'])
        trainIou.append(log['iou_score'])

    for log in test_logs_list:
        testDice.append(log['dice_loss'])
        testIou.append(log['iou_score'])

    epochsList = np.arange(1, len(trainDice) + 1, 1)
    plotResults(args, epochsList, trainDice, testDice, "Train Dice Loss", "Test Dice Loss", "Epochs", "Dice Loss",
                "Dice Loss for DeepLabV3 pretrained on " + args.arch)
    plotResults(args, epochsList, trainIou, testIou, "Train IoU score", "Test IoU score", "Epochs", "IoU Score",
                "IoU Score for DeepLabV3 pretrained on " + args.arch)


def plotResults(args, xList, y1List, y2List, y1Label, y2Label, xAxisLabel, yAxisLabel, title):
    plt.plot(xList, y1List, label=y1Label)
    plt.plot(xList, y2List, label=y2Label)
    plt.legend()
    plt.xlabel(xAxisLabel)
    plt.ylabel(yAxisLabel)
    plt.title(title)
    imageName = title.replace(" ", "_") + ".png"
    imagePath = os.path.join(args.output_path, imageName)
    plt.savefig(imagePath)