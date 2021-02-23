## Fair Empirical Risk Minimization Via Exponential Rényi Mutual Information
This repository is dedicated to implementation of Fair Empirical Risk Minimization (FERMI) algorithms proposed in . FERMI provides the first ****stochastic**** algorithm with theroretical convergence guarantees for promoting fairness in classification tasks. To balance the accuracy and fairness, **fair risk minimization through exponential Rényi mutual information** framework minimizes the following objective function:

<div align='center'> 
<img src="General_Framework.png" width="750" align='center'>
</div>

where the first term represents the population risk (accuracy) and the second term is a regularizer promoting exponential Rényi mutual information (ERMI) between the sensitive attribute(s) and predictions. Note that ERMI is a stronger notion of fairness compared to existing notions of fairness such as mutual information [Kamishima et al., 2011, Rezaei et al., 2020, Steinberget al., 2020, Zhang et al., 2018, Cho et al., 2020a], Pearson correlation [Zafar et al., 2017], false positive/negative rates[Bechavod and Ligett, 2017], Hilbert Schmidt independence criterion (HSIC) [Pérez-Suay et al., 2017], and Rényicorrelation [Baharlouei et al., 2020, Grari et al., 2020, 2019], in the sense that it upper bounds all aforementioned notions. Thus, minimizing ERMI guarantees the fairness of model under those notions. Below table demonstrates the capabilities of FERMI over state-of-the-art approaches in the literature.


**Method** | **Type** | **Constant Memory** | **Problem Agnostic** | **On Pre-trained** | **Unlabeled Data** | **Adaptive** | **Consistency**
--- | --- | --- | --- | --- | --- | --- | ---
LwF | Data | :heavy_check_mark: | :x: | :heavy_check_mark: | :x: | :x: | :x:
EBLL | Data | :x: | :x: | :x: | :x: | :x: | :x:
EWC | Model | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x:
IMM | Model | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: | :x:
SI | Model | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x: | :x:
MAS | Model | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:
Alpha | Model | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:


## Dependencies


## Binary Classification with Binary Sensitive Attribute 
To run the code for a binary classification problem with a binary sensitive attribute use the following command:  

```
python BinaryClassification/Binary_FERMI.py 
```


## Stochastic FERMI for Large-scale Neural Networks on Datasets with Multiple Sensitive Attributes
The implementation of Algorithm 1 in , specialized to a 4-layer neural network on color mnist dataset can be found in NeuralNetworkMnist folder. You can run it on color mnist dataset via:

```
python NeuralNetworkMnist/code_cm.py 
```

