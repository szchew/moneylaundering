import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import roc_curve, auc, make_scorer, average_precision_score
from sklearn.metrics import precision_recall_curve

def processing(feature_x):
    category_feature = feature_x[["Receiving Currency", "Payment Currency", "Payment Format"]]
    le = LabelEncoder()
    cf_copy = category_feature.copy()
    cf_copy["Receiving Currency"] = le.fit_transform(category_feature["Receiving Currency"])
    cf_copy["Payment Currency"] = le.fit_transform(category_feature["Payment Currency"])
    cf_copy["Payment Format"] = le.fit_transform(category_feature["Payment Format"])
    # print(category_feature)
    
    ohe = OneHotEncoder(categories = 'auto').fit(cf_copy)
    onehot = pd.DataFrame(ohe.transform(cf_copy).toarray())
    
    numeric = pd.DataFrame(feature_x[["Amount Received","Amount Paid"]]).reset_index()

    res = pd.concat([numeric, onehot], axis=1).iloc[:,1:]

    return res

def merge():                             
    df1 = pd.read_csv("LI-Small_Trans.csv")   
    df2 = pd.read_csv("HI-Small_Trans.csv")
    
    df1 = pd.concat([df1,df2],axis=0,ignore_index=True)   
      
    df1 = df1.reset_index(drop=True)      
    df1.to_csv('total.csv')  


raw_dat = pd.read_csv("HI-small_Trans.csv")
origin = shuffle(raw_dat)
origin = origin.iloc[:, 5:11]

label = origin['Is Laundering']
data = origin.iloc[:,:5]

processed_data = processing(data)
processed_data.columns = processed_data.columns.astype(str)

zeros = processed_data[label == 0]
ones = processed_data[label > 0]
zerolabel = label[label == 0]
onelabel = label[label > 0]

train_0,test_0,train_0l,test_0l = train_test_split(zeros, zerolabel, test_size=0.996)
final_test, xx, final_label,yy = train_test_split(test_0, test_0l, test_size=0.99996)
train_1,test_1,train_1l,test_1l = train_test_split(ones, onelabel, test_size=0.01)

data_train = pd.concat([train_0, train_1], axis=0)
label_train = pd.concat([train_0l, train_1l], axis=0)
data_test = pd.concat([test_1, final_test], axis=0)
label_test = pd.concat([test_1l, final_label], axis=0)

param = {"activation" : ["identity", "logistic", "tanh", "relu"], 
            "solver" : ["lbfgs", "sgd", "adam"],
            "alpha":[0.0001,0.001,0.00005],
            "learning_rate":["constant", "invscaling", "adaptive"],
            "max_iter":[500]}

mx = MLPClassifier() 

grid_search = GridSearchCV(mx, param, cv=10, scoring = make_scorer(average_precision_score, greater_is_better=True))
grid_search.fit(data_train,label_train)

print(grid_search.best_params_)
print("--------------------------------")

cur_score = grid_search.best_estimator_.score(data_train,label_train)
print(cur_score)
test_score = grid_search.best_estimator_.score(data_test,label_test)
print(test_score)


# y_pred = mx.predict(X_test)
# acc_score = accuracy_score(y_pred, y_test)
# print('mse_score',acc_score)
