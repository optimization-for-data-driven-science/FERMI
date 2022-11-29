## A Stochastic Optimization Framework for Fair Risk Minimization
This repository presents the implementation of Fair Empirical Risk Minimization via Exponential Renyi Mutual Information (FERMI) proposed in [A Stochastic Optimization Framework for Fair Risk Minimization](https://arxiv.org/abs/2102.12586) paper. FERMI provides the first **stochastic** algorithm with theoretical convergence guarantees to optimal points (tradeoff between accuracy and fairness) for promoting fairness in classification tasks. To balance accuracy and fairness, our framework minimizes the following objective function:

<div align='center'> 
<img src="General_Framework.png" width="750" align='center'>
</div>

where the first term represents the population risk (accuracy) and the second term is a regularizer promoting exponential Rényi mutual information (ERMI) between the sensitive attribute(s) and predictions. Note that ERMI is a stronger notion of fairness compared to existing notions of fairness such as mutual information [Kamishima et al., 2011, Rezaei et al., 2020, Steinberget al., 2020, Zhang et al., 2018, Cho et al., 2020a], Pearson correlation [Zafar et al., 2017], false positive/negative rates[Bechavod and Ligett, 2017], Hilbert Schmidt independence criterion (HSIC) [Pérez-Suay et al., 2017], and Rényicorrelation [Baharlouei et al., 2020, Grari et al., 2020, 2019], in the sense that it upper bounds all aforementioned notions. Thus, minimizing ERMI guarantees the fairness of model under those notions. In the following table we compare FERMI with several state-of-the-art approaches in the literature. Note that the abbrevations NB, Cont., Stoch., DP, EOD, MI and RC stand for Non-binary, Continuous, Stochastic, Dempographic Parity, Equalized Odds, Mutual Information, and Rényi Correlation respectively. 


**Reference** | **NB Target** | **NB Attribute** | **NB Code** | **Beyond Logistic** | **Unbiased Stoch. Alg** | **Convergence Guarantee**
:-: | :-: | :-: | :-: | :-: | :-: | :-:
Sotchastic FERMI | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | O(ε<sup>-4</sup>)
Cho et al. [2020a] | :heavy_check_mark: | :heavy_check_mark: | :x: | :heavy_check_mark: | :heavy_check_mark: | :x:
Cho et al. [2020b] | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | biased | :x:
Baharlouei et al. [2020] | :heavy_check_mark: | :heavy_check_mark: | :x: | :heavy_check_mark: | :x: | :heavy_check_mark:
Rezaei et al. [2020] | :x: |  :x: | :x: | :x: | :x: | :x:
Jiang et al. [2020] | :x: | :heavy_check_mark: | :x: | :x: | :x: | :x:
Donini et al. [2018] | :x: | :heavy_check_mark: | :x: | EOD | :x: | :x:
Zhang et al. [2018] | :heavy_check_mark: | :heavy_check_mark: | :x: | DP, EOD | biased | :x:



## Dependencies
The following packages must be installed via Anaconda or pip before running the codes. Download and install **Python 3.x version** from [Python 3.x Version](https://www.python.org/downloads/):
Then install the following packages via Conda or pip:
* [Numpy](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
* [Pandas](https://anaconda.org/conda-forge/matplotlib)
* [Scikit learn](https://scikit-learn.org/stable/install.html)
* [Matplotlib](https://matplotlib.org/stable/users/installing.html)
* [PyTorch](https://pytorch.org/get-started/locally/)

## Install FERMI as a Python Package
FERMI package is publicly available at [FERMI-ODDS](https://pypi.org/project/FERMI-ODDS/): You can install FERMI via pip using the following command:

```
pip install FERMI-ODDS
```

## Binary Classification with Binary Sensitive Attribute (Adult Dataset)
To run the code for a binary classification problem with a binary sensitive attribute on the Adult dataset, run the following commands:

```
from FERMI import AdultDataset
from FERMI import FERMIBinary

AdultDataset.download_data()
X_train, S_train, Y_train = AdultDataset.read_binary_adult(mode='train')
X_test, S_test, Y_test = AdultDataset.read_binary_adult(mode='test')


fermi_instance = FERMIBinary.FERMI(X_train, X_test, Y_train, Y_test, S_train, S_test)
FERMIBinary.FERMI_Logistic_Regression(fermi_instance)
```

The above code updates the parameters of a logistic regression model via stochastic gradient descent algorithm. FERMI_Logistic_Regression has multiple parameters including batch_size (default=64), initial epochs (default = 300, in this phase we learn the model without the fairness regularizer), epochs (default=1000, total number of epochs), etc. 

### Fair Classification in the Presence of Missing Sensitive Attributes (Adult Dataset)
If the sensitive attributes are not fully available, FERMI still provides an accuracy-bias tradeoff without removing the data entries containing missing values.

```
from FERMI import AdultDataset
from FERMI import FERMIBinary

AdultDataset.download_data()
X_train, S_train, Y_train = AdultDataset.read_binary_adult(mode='train')
X_test, S_test, Y_test = AdultDataset.read_binary_adult(mode='test')


fermi_instance = FERMIBinary.FERMI(X_train, X_test, Y_train, Y_test, S_train, S_test)
FERMIBinary.FERMI_Logistic_Regression(fermi_instance, **is_missing=true**)
```

To use FERMI on other datasets, please create X_train, X_test, S_train, S_test, Y_train, and Y_test variables and use the code above instead of Adult dataset.

## Non-Binary Fair Classification (Adult Dataset)
To run the code for a classification problem with non-binary sensitive attribute on the Adult dataset, run the following commands:

```
from FERMI import AdultDataset
from FERMI import FERMIBinary

AdultDataset.download_data()
X_train, S_train, Y_train = AdultDataset.read_non_binary_adult(mode='train')
X_test, S_test, Y_test = AdultDataset.read_non_binary_adult(mode='test')

fermi_instance = FERMIDiscrete.FERMI(X_train, X_test, Y_train, Y_test, S_train, S_test)
FERMIDiscrete.FERMI_Logistic_Regression(fermi_instance)
```



## Stochastic FERMI for Large-scale Neural Networks on Datasets with Multiple Sensitive Attributes (Non-binary Labels and Sensitive Attributes)
The implementation of Algorithm 1 in [paper](https://arxiv.org/abs/2102.12586), specialized to a 4-layer neural network on color mnist dataset can be found in NeuralNetworkMnist folder. You can run it on color mnist dataset via:

```
python NeuralNetworkMnist/code_cm.py 
```


## Fair Toxic Comment Classification
We apply FERMI to the Toxic Comment Classification dataset where the underlying task is to predict whether a given published comment in social media is toxic. The sensitive attribute is religion that is binarized into two groups: Christians in one group; Muslims and Jews in the other group. Training a
neural network without considering fairness leads to higher false positive rate for the Jew-Muslim group. FERMI reduces the false positive rate gap between two
religious groups. Please see the notebook Toxic_Comment.ipynb notebook for the implementation.
