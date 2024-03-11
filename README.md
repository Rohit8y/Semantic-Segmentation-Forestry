# Satellite Semantic Segmentation Forestry


### [**Contents**](#)
1. [Description](#descr)
2. [Installation](#install)
3. [Data Preparation](#prepare)
4. [Training](#train)
5. [Metrics](#metrics)
6. [References](#ref)

---

### [**Description**](#) <a name="descr"></a>
Semantic segmentation on satellite images of forests can identify and delineate the specific areas covered by trees, providing a detailed understanding of the distribution and extent of forested regions. This information is crucial not only for applications such as monitoring deforestation and assessing forest health, but also for detecting and mapping forest fires, enabling timely response and mitigation efforts.

We utilized the Göttingen aerial dataset for our experiments. However, this dataset has a limitation of containing only 38 labeled images. Due to the shortage of training data, we decided to leverage transfer learning by employing the ImageNet dataset[[1]](#1) for pre-training. Since the segmentation model consists of an encoder and a decoder, we used a pre-trained ResNet as the encoder and relied on the DeepLabV3Plus decoder[[2]](#2) architecture. Given the scarcity of the training set, we performed end-to-end fine-tuning to optimize the model. The evaluation of the models was carried out using the Intersection over Union (IoU) metric, which measures the overlap between the predicted and ground truth segmentation masks. A sample result showcasing the performance of our best model is illustrated below.

![Resnet152_results](https://user-images.githubusercontent.com/38680205/229292089-6c84c8f6-0cf5-4cab-aea0-45cecbc77cb4.png)


---

### [**Installation**](#) <a name="install"></a>

**1.** Clone the repository:

``` shell
git clone git@github.com:Rohit8y/Financial-Time-Series.git
cd Financial-Time-Series
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



---

### [**References**](#) <a name="ref"></a>

<a id="1">[1]</a> 
Russakovsky, Olga, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang et al. "Imagenet large scale visual recognition challenge." *International journal of computer vision 115 (2015): 211-252.*

<a id="2">[2]</a> 
Chen, Liang-Chieh, George Papandreou, Florian Schroff, and Hartwig Adam. "Rethinking atrous convolution for semantic image segmentation." arXiv preprint arXiv:1706.05587 (2017).



