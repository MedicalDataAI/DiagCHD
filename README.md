<div align="center">
<h1><img src='/sundry/1f52c.gif' width="35px"> DiagCHD <img src='/sundry/1f4a1.gif' width="35px"></h1>
<h3>Deep Learning-Driven Multi-Task Model for Auto-segmentation and Classification of Ultrasound Diagnosis of Fetal Congenital Heart Disease</h3>



<!--
### [Project Page]() | [Paper link]()
-->

</div>

## <img src='/sundry/1f43c.gif' width="30px"> News

* **`March 27, 2024`:** We released our inference code. Paper/Project pages are coming soon. Please stay tuned! <img src='/sundry/1f379.gif' width="25px">

## <img src='/sundry/1f4f0.gif' width="30px"> Abstract
Objective: This study aimed to evaluate the performance of deep learning (DL)-based automatic segmentation and classification of ultrasound images in diagnosing fetal congenital heart disease (CHD).
Methods: Ultrasound image of the fetal heart's four-chamber view and maternal clinical information were retrospectively collected from 736 pregnant women. A DL model was utilized to segment and classify the images, with comparisons of the established DL-based models and experts’ diagnosis. The classification performance was evaluated using a confusion matrix to calculate the accuracy (ACC), recall, F1 score and precision. In the segmentation task, Categorical Crossentropy and Dice Loss were used as loss functions, with the Dice coefficient and mean Intersection over Union (mean-IOU) as evaluation metrics.
Results: The proposed multi-task DL model in this study demonstrated superior performance in the segmentation of the four-chamber view of the fetal heart with mean-IOU of 0.832, outperforming other models. In the classification of fetal CHD, the model in the training dataset achieved the accuracy, precision, and AUC values of 96.23%, 100%, and 96.75%, respectively. In the internal validation dataset, the DL algorithm achieved a precision of 85.71%, compared to an average diagnostic precision of 73.38% by experts, indicating robust performance.
Conclusion: The multi-task DL proposed demonstrated outstanding performance in both the segmentation and classification of fetal heart ultrasound images, supporting the use of DL techniques for accurate and rapid CHD diagnosis. This model promises to enhance clinical practice and improve the management and treatment of fetal CHD patients.


## <img src='/sundry/1f9f3.gif' width="30px"> Environment Setups

* python 3.8
* cudatoolkit 11.8.0
* cudnn 8.1.0.77
* See 'requirements.txt' for Python libraries required

```shell
conda create -n DiagCHD python=3.8
conda activate DiagCHD
conda install cudatoolkit=11.8.1 cudnn=8.1.0.77
pip install tensorflow-gpu==2.8.0
# cd /xx/xx/DiagCHD
pip install -r requirements.txt
```


## <img src='/sundry/1f5c2-fe0f.gif' width="30px"> Model Checkpoints
Download the zip of [model checkpoints](https://pan.baidu.com/s/1NBT27balyJLss3iykFEuLg?pwd=xdjw) (key:```xdjw```), decompress and put all pkl files into ../DiagCHD/weights/checkpoints.

## <img src='/sundry/听诊器.gif' width="30px"> Our Dataset 
We have provided sample data in the dataset directory.

## <img src='/sundry/1f3ac.gif' width="30px"> Inference Demo
You can visualize our inference results and evaluation metrics using the following commands.:

*  inference
```shell
python test.py 

```




<!--
## Acknowledgement


## Citation
-->
