## Fair Empirical Risk Minimization Via Exponential Rényi Mutual Information
This repository is dedicated to implementation of Fair Empirical Risk Minimization (FERMI) algorithms proposed in . FERMI provides the first ****stochastic**** algorithm with theroretical convergence guarantees for promoting fairness in classification tasks. To balance the accuracy and fairness, **fair risk minimization through exponential Rényi mutual information** framework minimizes the following objective function:

<div align='center'> 
<img src="General_Framework.png" width="750" align='center'>
</div>

where the first term represents the population risk (accuracy) and the second term is a regularizer promoting exponential Rényi mutual information (ERMI) between the sensitive attribute(s) and predictions. Note that ERMI is a stronger notion of fairness compared to existing notions of fairness such as mutual information [Kamishima et al., 2011, Rezaei et al., 2020, Steinberget al., 2020, Zhang et al., 2018, Cho et al., 2020a], Pearson correlation [Zafar et al., 2017], false positive/negative rates[Bechavod and Ligett, 2017], Hilbert Schmidt independence criterion (HSIC) [Pérez-Suay et al., 2017], and Rényicorrelation [Baharlouei et al., 2020, Grari et al., 2020, 2019], in the sense that it upper bounds all aforementioned notions. Thus, minimizing ERMI guarantees the fairness of model under those notions. Below table demonstrates the capabilities of FERMI over state-of-the-art approaches in the literature.


**Reference** | **NB Target** | **NB Attribute** | **NB both exp.** | **Cont. Target** | **Violation Notion** | **Unbiased Stoch. Alg** | **Convergence Guarantee**
:-: | :-: | :-: | :-: | :-: | :-: | :-: | :-:
Sotchastic FERMI | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | dp, eod, MI, RC, ERMI | :x: | :x:
Batch FERMI | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | dp, eod, MI, RC, ERMI | :x: | :x:
Cho et al. [2020a] | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | dp, eod, MI | :x: | :x:
Cho et al. [2020b] | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | dp, eod | :x: | :x:
Baharlouei et al. [2020] | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | dp, eod, RC | :x: | :x:
Rezaei et al. [2020] | :x: |  :x: | :x: | :x: | dp, eod | :heavy_check_mark: | :x:
Jiang et al. [2020] | :x: | :heavy_check_mark: | :x: | :x: | dp | :heavy_check_mark: | :heavy_check_mark:
Donini et al. [2018] | :x: | :heavy_check_mark: | :x: | :x: | eod | :heavy_check_mark: | :heavy_check_mark:
Zhang et al. [2018] | :heavy_check_mark: | :heavy_check_mark: | :x: | :heavy_check_mark: | dp, eod | :heavy_check_mark: | :heavy_check_mark:

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

