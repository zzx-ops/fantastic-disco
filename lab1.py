import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import time

col_names=['gameDuration','seasonId','winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']

pima= pd.read_csv("new_data.csv",header=None, names=col_names) 
test= pd.read_csv("test_set.csv",header=None, names=col_names) 

pima=pima.iloc[1:]
test=test.iloc[1:]

cols=['gameDuration','seasonId','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
cols_2=['firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald']

pima[cols]=pima[cols].apply(pd.to_numeric,errors='coerce')
test[cols]=test[cols].apply(pd.to_numeric,errors='coerce')           


#pima=pima[pima['gameDuration']>300]


pima[cols_2]=pima[cols_2].replace([2],[-1])#修改first类的值 简单的线性划分 所以分为正负效果会更好
test[cols_2]=test[cols_2].replace([2],[-1])

#增加特征值
pima.insert(loc=17,column='d_towerkill',value=(pima.t1_towerKills)-(pima.t2_towerKills))
test.insert(loc=17,column='d_towerkill',value=(test.t1_towerKills)-(test.t2_towerKills))

pima.insert(loc=18,column='d_inhibitorkills',value=(pima.t1_inhibitorKills)-(pima.t2_inhibitorKills))
test.insert(loc=18,column='d_inhibitorkills',value=(test.t1_inhibitorKills)-(test.t2_inhibitorKills))

pima.insert(loc=19,column='d_baronkills',value=(pima.t1_baronKills)-(pima.t2_baronKills))
test.insert(loc=19,column='d_baronkills',value=(test.t1_baronKills)-(test.t2_baronKills))

pima.insert(loc=20,column='d_dragonkills',value=(pima.t1_dragonKills)-(pima.t2_dragonKills))
test.insert(loc=20,column='d_dragonkills',value=(test.t1_dragonKills)-(test.t2_dragonKills))

pima.insert(loc=21,column='d_riftheraldkill',value=(pima.t1_riftHeraldKills)-(pima.t2_riftHeraldKills))
test.insert(loc=21,column='d_riftheraldkill',value=(test.t1_riftHeraldKills)-(test.t2_riftHeraldKills))


#选择不同的特征值进行训练

feature_cols=['firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','d_towerkill','d_inhibitorkills','d_baronkills','d_dragonkills','d_riftheraldkill']

#feature_cols=['firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills','d_towerkill','d_inhibitorkills','d_baronkills','d_dragonkills','d_riftheraldkill']

#feature_cols=['firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']


#训练

'''
def cv_score(d):#该函数用于遍历求最优值
    clf = DecisionTreeClassifier(criterion="gini",splitter="best",max_depth=14,min_samples_leaf=6)
    clf.fit(X_train, y_train)
    return(clf.score(X_train, y_train), clf.score(X_test, y_test))
'''

start = time.time()

test_x=test[feature_cols]

X=pima[feature_cols]
y=pima.winner

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.7,random_state=1)

clf=DecisionTreeClassifier(criterion="gini",splitter="best",max_depth=14,min_samples_leaf=6,max_features=10)
clf=clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)


print("Train Accuracy:",accuracy_score(y_test,y_pred))

test_pred=clf.predict(test_x)

print("Test Accuracy:",accuracy_score(test.winner,test_pred))

scores = cross_val_score(clf, test_x, test.winner, cv=10)
scores=  scores.sum()/10

print("cross score:",scores)
end = time.time()
print(end-start,"s")
'''
depths = np.arange(1,20)
scores = [cv_score(d) for d in depths]
tr_scores = [s[0] for s in scores]#训练集分数
te_scores = [s[1] for s in scores]#测试集分数

# 找出交叉验证数据集评分最高的索引
tr_best_index = np.argmax(tr_scores)
te_best_index = np.argmax(te_scores)

print("bestdepth:", te_best_index+1, " bestdepth_score:", te_scores[te_best_index], '\n')

'''


#visualizing
from six import StringIO
from IPython.display import Image 
from sklearn.tree import export_graphviz
import pydotplus
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
dot_data =StringIO()



export_graphviz(clf, out_file=dot_data,
     filled=True, rounded=True,
     special_characters=True, feature_names =feature_cols,
     class_names=['1','2']
)


graph =pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())



#display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Mpimg is used to read pictures
diabetes = mpimg.imread('diabetes.png') 
# Diabetes is already an np.array and can be processed at will
plt.imshow(diabetes) # Show Picture
plt.axis('off') # Do not show axes
plt.show()
