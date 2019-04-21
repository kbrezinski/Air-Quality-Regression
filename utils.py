
import pandas as pd
import numpy as np

def read_data(file):
    data = pd.ExcelFile(file).parse()
    data = data.iloc[:,2:].astype(np.float32)
    data = data.mask(data == -200)
    return data
