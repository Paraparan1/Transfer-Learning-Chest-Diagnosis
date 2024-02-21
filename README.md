# Transfer-Learning-Chest-Diagnosis
Enhancing Chest Diagnosis Through Transfer Learning Techniques

Abstract: 
In the rapidly advancing domain of medical imaging, chest X-ray diagnosis is paramount for the identification of pulmonary diseases. This research addresses the challenge of heightening the classification accuracy of such diseases by harnessing the capabilities of transfer learning techniques. Recognising the limitations of traditional methodologies and the transformative potential of deep learning, the study's approach utilised prominent models such as ResNet-50, InceptionV3, and Xception. It delved into the advantages of leveraging pre-trained weights from benchmark datasets like ImageNet, the potential of domain adaptation across related medical realms, the intricacies of fine-tuning procedures, and the influence of complex architectural designs on performance. Through methodical experimentation across diverse medical datasets, it was observed that models employing transfer learning significantly outperformed or were on par with their non-pre-trained counterparts across multiple metrics. Importantly, the research emphasises the profound need to tackle challenges presented by class imbalances and dataset specificities. The findings underscore the pivotal role of transfer learning in advancing chest imaging diagnosis, paving the way for future innovations in medical image analysis.
The following datasets were used:

Dataset 1 (Pneumonia): https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 
Dataset 2 (Tuberculosis): https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
Dataset 3 (NIH): https://www.kaggle.com/datasets/nih-chest-xrays/data
Dataset 4 (NIH_Balanced): https://www.kaggle.com/datasets/nih-chest-xrays/data
______________________________________________________________________________________________________________________
The dataset loaders featured the code that was used to load in the datasets and pre process the datasets
Dataset Loader Code files

Dataset 1 loader: Pneumonia_Kaggle_Dataset_Loader

Dataset 2 loader: Tuberculosis_Kaggle_Dataset_Loader

Dataset 3 loader: NIH_Data_Loader 

Dataset 4 loader: NIH_Class_Balanced_Loader (Subset of Dataset 3 with each class containing 800 images) 
______________________________________________________________________________________________________________________
The code files present in the file are seperated into two categories: Used for Dataset 1 and 2 and Used for Dataset 3 
For no weights the weights has to be changed from weights='imagenet' to weights=None. 

Model Code Files for Dataset 1(Pneumonia) and 2 (Tuberculosis): 

CNN_Kaggle, Resnet50_Kaggle, Inceptionv3_Kaggle, Xception_Kaggle 


For the pneumonina and Tuberculosis datasets: 
class_names variable within the code needs to be changed:

Dataset 1(Pneumonia) - class_names = ['NORMAL', 'PNEUMONIA']

Dataset 2(Tuberculosis) - class_names = ['TUBERCULOSIS','NORMAL']


Model Code Files for Dataset3(NIH): CNN_nih, Resnet50_nih, Inception_nih, Xception_nih

---------------------------------------------------------------------------------------------------------------------
This file was used for the domain adaptation experiment:

domain_adaptation

The class names also need to be set for each respective model:
Dataset 1(Pneumonia) - class_names = ['NORMAL', 'PNEUMONIA']
Dataset 2(Tuberculosis) - class_names = ['TUBERCULOSIS','NORMAL']

For the domain_adapatation the model = load_model(" ") needs to be set with each respective model:
inceptionv3_Tuberculosis_Kaggle_Model.h5
xception_pneumonia_Kaggle_Model.h5 
---------------------------------------------------------------------------------------------------------------------
Fine tuned code:

finetuned_xception_nih - Finetuning on unbalanced data. 

finetuned_xception_balanced - Finetuning on balanced data. 

This model is loaded in: 
xception_NIH_Model.h5 

The adaptive learning rate callback was not used for Dataset 3 experiments. 

The following learning rate parameters was tested for Dataset 3: 
0.00001
0.0001
---------------------------------------------------------------------------------------------------------------------

Complex Architecture - This model contains the xception model with added layers:

xception_complex 

This can be run without any alterations. 
______________________________________________________________________________________________________________________

EDA code files:

EDA_Pneumonia_Kaggle
EDA_Tuberculosis_Kaggle
EDA_NIH

These files show sample images of each dataset. They are run from the orginal dataset and not the numpy array files.
______________________________________________________________________________________________________________________

