from aif360.datasets import BankDataset
from aif360.algorithms import inprocessing


df = BankDataset()
data = df.convert_to_dataframe()
print(data)
