import tushare as ts
import pandas as pd
import numpy as np
from sklearn import preprocessing

def creat_stock_csv(Target):
    DataSh = ts.get_hist_data('sh')
    DataSz = ts.get_hist_data('sz')
    DataZxb = ts.get_hist_data('zxb')
    DataCyb = ts.get_hist_data('cyb')
    DataZs = pd.concat([DataSh,DataSz,DataZxb,DataCyb],axis=1)
    Error_Data = []
    Max_len = 732
    for target in Target:
        Data=ts.get_hist_data(target, retry_count=5)
        DATA=pd.concat([Data,DataZs],axis=1)
        DATA=DATA.dropna(0)
        shape=np.shape(DATA)
        if shape[0] < 600 or shape[1] < 66:
            print(target+' error data shape'+str(list(np.shape(DATA))))
            Error_Data.append(target)
            continue
        if shape[0] > 732 : Max_len=shape[0]
        DATA.to_csv('./data/'+target+'.csv',index=False,header=False)
        print(np.shape(DATA))
    for target in Error_Data:
        Target.remove(target)
    return Max_len, shape[1]

def creat_stock_data(Target):
    output = []
    minmax_info = []
    sample = 0
    for target in Target:
        data = pd.read_csv('./data/'+target+'.csv')
        data = data.as_matrix().astype('float32')
        outdata = preprocessing.MinMaxScaler((0,1))
        outdata.fit(data)
        data = outdata.transform(data)
#         while np.shape(data)[0] < max_len-1:
#             data[np.shape(data)[0],:] = -1
# #        data = preprocessing.scale(np.array(data))
#        print(data)
        output.append(data)
        minmax_info.append(outdata)
        sample+=1
    print(np.shape(output))
#    output = np.array(output).reshape((-1, max_len-1, 66))
    return  output, minmax_info, sample

if __name__ == "__main__":
    Target = ['300024', '002230', '600460', '600570', '600000', '002415', '601318', '600028']
    Max_len, Shape = creat_stock_csv(Target)
    Output, Minmax_info, Sample = creat_stock_data(Target)
    print(np.shape(Output))
    print(Output)