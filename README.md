# ULTS
A unified and standardized library of unsupervised representation learning approaches for time series


## Description of this library:
ULTS is a unified and standardized library under the PyTorch framework to enable quick and convenient evaluations on unsupervised representation learning approaches for time series. ULTS integrates 17 representative models covering 2 deep clustering, 2 reconstruction-based and 13 self-supervised learning methods including 2 adversarial, 2 predictive and 9 contrastive ones. For more information, please refer to our paper:  [[Unsupervised Representation Learning for Time Series: A Review]](https://arxiv.org/abs/2308.01578).


## Abstract
Unsupervised representation learning approaches aim to learn discriminative feature representations from unlabeled data, without the requirement of annotating every sample. Enabling unsupervised representation learning is extremely crucial for time series data, due to its unique annotation bottleneck caused by its complex characteristics and lack of visual cues compared with other data modalities. In recent years, unsupervised representation learning techniques have advanced rapidly in various domains. However, there is a lack of systematic analysis of unsupervised representation learning approaches for time series. To fill the gap, we conduct a comprehensive literature review of existing rapidly evolving unsupervised representation learning approaches for time series. Moreover, we also develop a unified and standardized library, named ULTS ({i.e., Unsupervised Learning for Time Series), to facilitate fast implementations and unified evaluations on various models. With ULTS, we empirically evaluate state-of-the-art approaches, especially the rapidly evolving contrastive learning methods, on 9 diverse real-world datasets. We further discuss practical considerations as well as open research challenges on unsupervised representation learning for time series to facilitate future research in this field.

## Taxonomy:
![image](https://github.com/mqwfrog/ULTS/blob/main/taxonomy.png)


## Organization:
![image](https://github.com/mqwfrog/ULTS/blob/main/organization.png)
  
## Models Implemented in ULTS:
<table>
    <tr>
        <td>1st Category</td>
        <td>2nd Category</td>
        <td>3rd Category</td>
        <td>Model</td>
    </tr>
    <tr>
        <td rowspan="2">Deep Clustering Methods</td>
        <td>-</td>
        <td>-</td>
        <td>DeepCluster https://github.com/facebookresearch/deepcluster </td>
    </tr>
    <tr>
        <td>-</td>
        <td>-</td>
        <td>IDFD https://github.com/TTN-YKK/Clustering_friendly_representation_learning</td>
    </tr>
    <tr>
        <td rowspan="2">Reconstruction-based Methods</td>
        <td>-</td>
        <td>-</td>
        <td>TimeNet https://github.com/paudan/TimeNet</td>
    </tr>
    <tr>
        <td>-</td>
        <td>-</td>
        <td>Deconv https://github.com/cauchyturing/Deconv_SAX</td>
    </tr>
     <tr>
        <td rowspan="13">Self-supervised Learning Methods</td>
        <td>Adversarial</td>
        <td>-</td>
        <td>TimeGAN https://github.com/jsyoon0823/TimeGAN<br> TimeVAE https://github.com/abudesai/timeVAE</td>
    </tr>
    <tr>
        <td>Predictive</td>
        <td>-</td>
        <td>EEG-SSL https://github.com/mlberkeley/eeg-ssl <br> TST https://github.com/gzerveas/mvts_transformer</td>
    </tr>
     <tr>
         <td rowspan="3">Contrastive</td>
        <td>Instance-Level</td>
        <td>SimCLR https://github.com/google-research/simclr<br> BYOL https://github.com/deepmind/deepmind-research/tree/master/byol<br> CPC https://github.com/facebookresearch/CPC_audio</td>
    </tr>
    <tr>
        <td>Prototype-Level</td>
        <td>SwAV https://github.com/facebookresearch/swav<br> PCL https://github.com/salesforce/PCL<br> MHCCL https://github.com/mqwfrog/MHCCL</td> 
    </tr>
    <tr>
        <td>Temporal-Level</td>
        <td>TS2Vec https://github.com/yuezhihan/ts2vec<br> TS-TCC https://github.com/emadeldeen24/TS-TCC<br> T-Loss https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries</td>
    </tr>
</table>

## Requirements for this library:
- Python ≥ 3.6
- PyTorch ≥ 1.4

## Required packages for this library:
- numpy
- sklearn
- openpyxl 
- torchvision
- random
- copy
- pandas
- matplotlib
- time
- collections
- scipy
- pynndescent
- builtins
- math
- shutil
- os
- sys
- warnings
- tqdm
- argparse
- tensorboard_logger 


## Data:
- The [UCI](https://archive.ics.uci.edu/datasets) archive includes 85 multivariate time series datasets for classification tasks. These datasets covers various application fields including audio spectra classification, business, ECG/EEG classification, human activity recognition, gas detection, motion classification, etc.
- The [UEA](http://www.timeseriesclassification.com/dataset.php) archive includes 30 multivariate time series datasets, covers the application fields of audio spectra classification, ECG/EEG/MEG classification, human activity recognition, motion classification, etc.
- The [MTS](http://www.mustafabaydogan.com/multivariate-time-series-discretization-for-classification.html) archive, also known as Baydogan's archive, includes 13 multivariate time series datasets, covers the application fields of audio spectra classification, ECG classification, human activity recognition, motion classification, etc. 


## Codes:
The codes in ULTS library are organized as follows:
- The [main.py](https://github.com/mqwfrog/ULTS/blob/main/main.py) includes the training method for all models.
- The [models](https://github.com/mqwfrog/ULTS/tree/main/models) folder contain all 17 unsupervised learning models.
- The [data_preprocess](https://github.com/mqwfrog/ULTS/tree/main/data_preprocess) folder contain the codes to preprocess data from different archives.
- The [data_loader](https://github.com/mqwfrog/ULTS/tree/main/data_loader) folder contains the codes to perform augmentation transformations and to load the dataset.


## Running:
<pre>
python main.py \
--dataset_name wisdm \
--uid SimCLR
--lr 0.03 \
--batch_size 128 \
--feature_size 128
</pre>



## Results:
- The experimental results will be saved in "experiment_{args.model}_{args.dataset}" directory by default.


## Citation:
If you find any of the codes helpful, kindly cite our paper.   

<pre>
@misc{meng2023unsupervised, 
      title={Unsupervised Representation Learning for Time Series: A Review}, 
      author={Qianwen Meng and Hangwei Qian and Yong Liu and Yonghui Xu and Zhiqi Shen and Lizhen Cui},   
      year={2023},
      eprint={2308.01578},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
</pre>

## References:
Part of the codes are referenced from  
https://github.com/PatrickHua/SimSiam
https://github.com/facebookresearch/deepcluster
https://github.com/TTN-YKK/Clustering_friendly_representation_learning
https://github.com/paudan/TimeNet
https://github.com/cauchyturing/Deconv_SAX
https://github.com/jsyoon0823/TimeGAN
https://github.com/abudesai/timeVAE
https://github.com/joergsimon/SSL-ECG-Paper-Reimplementaton
https://github.com/mlberkeley/eeg-ssl
https://github.com/gzerveas/mvts_transformer
https://github.com/google-research/simclr
https://github.com/deepmind/deepmind-research/tree/master/byol
https://github.com/facebookresearch/CPC_audio
https://github.com/abhinavagarwalla/swav-cifar10
https://github.com/facebookresearch/swav
https://github.com/salesforce/PCL
https://github.com/mqwfrog/MHCCL
https://github.com/yuezhihan/ts2vec
https://github.com/emadeldeen24/TS-TCC
https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
