# Satellite Semantic Segmentation Forestry


### [**Contents**](#)
1. [Description](#descr)
2. [Installation](#install)
3. [Data Preparation](#prepare)
4. [Training](#train)
5. [References](#ref)

---

### [**Description**](#) <a name="descr"></a>
Semantic segmentation on satellite images of forests can identify and delineate the specific areas covered by trees, providing a detailed understanding of the distribution and extent of forested regions. This information is crucial not only for applications such as monitoring deforestation and assessing forest health, but also for detecting and mapping forest fires, enabling timely response and mitigation efforts.

We utilized the Göttingen aerial dataset for our experiments. However, this dataset has a limitation of containing only 38 labeled images. Due to the shortage of training data, we decided to leverage transfer learning by employing the ImageNet dataset[[1]](#1) for pre-training. Since the segmentation model consists of an encoder and a decoder, we used a pre-trained ResNet as the encoder and relied on the DeepLabV3Plus decoder[[2]](#2) architecture. Given the scarcity of the training set, we performed end-to-end fine-tuning to optimize the model. The evaluation of the models was carried out using the Intersection over Union (IoU) metric, which measures the overlap between the predicted and ground truth segmentation masks. A sample result showcasing the performance of our best model is illustrated below.

![Resnet152_results](https://user-images.githubusercontent.com/38680205/229292089-6c84c8f6-0cf5-4cab-aea0-45cecbc77cb4.png)


---

### [**Installation**](#) <a name="install"></a>

**1.** Clone the repository:

``` shell
$ git clone git@github.com:Rohit8y/Satellite-Segmentation-Forestry.git
$ cd Satellite-Segmentation-Forestry
```

**2.** Create a new Python environment and activate it:

``` shell
$ python3 -m venv py_env
$ source py_env/bin/activate
```

**3.** Install necessary packages:

``` shell
$ pip install -r requirements.txt
```

---

### [***Data Preparation***](#) <a name="prepare"></a>
The Göttingen dataset is partitioned into training and test sets following an 80:20 ratio. To ensure seamless integration of the data format with the code, we can utilize the script [generate_dataset.sh](https://github.com/Rohit8y/Satellite-Segmentation-Forestry/blob/main/generate_dataset.sh). This script automates the process of downloading the datasets and organizing them within the designated data folder.

**1.** Give execution permission to the script:

```
$ cd Satellite-Segmentation-Forestry
$ chmod 777 generate_dataset.sh
```

**2.** Run the script:

```
$ ./generate_dataset.sh
```
---

### [***Training***](#) <a name="train"></a>

The fine-tuning of pre-trained models for the segmentation task is accomplished using the main.py script. This script allows you to configure various parameters, load your dataset, and initiate the fine-tuning process to adapt the pre-trained models. After fine-tuning, the script saves the optimized model weights for deployment on new, unseen images

```
python main.py -h

usage: main.py [-h] [--data_path PATH] [--output_path OUTPUT] [--arch ARCH] [--epochs EPOCHS] [--lr LR]
               [--batch-size BS] [--wd WD] [--optimizer OPT] [--momentum M]
usage options:
  --help                show this help message and exit
  --data_path           the path pointing the dataset generated using generate_dataset.sh script
  --output_path         output directory for all models and plots
  --arch                architecture of the pre-trained encoder: [resnet 18|34|50|101|152]
  --epochs              number of total epochs to run (default: 50)
  --lr                  initial learning rate (default: 0.00005)
  --batch-size          mini-batch size (default: 2)
  --wd                  weight decay (default: 0.001)
  --optimizer           optimizer for updating the weights of the model: [sgd,adam]
  --momentum            momentum value in case of sgd optimizer

---

### [**References**](#) <a name="ref"></a>

<a id="1">[1]</a> 
Russakovsky, Olga, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang et al. "Imagenet large scale visual recognition challenge." *International journal of computer vision 115 (2015): 211-252.*

<a id="2">[2]</a> 
Chen, Liang-Chieh, George Papandreou, Florian Schroff, and Hartwig Adam. "Rethinking atrous convolution for semantic image segmentation." arXiv preprint arXiv:1706.05587 (2017).



