import os
import sklearn.svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def get_real_data(strPath):
    dat=[]
    with open(strPath, 'r') as rf:
        for line in rf.readlines():
            dat.append([float(v) for v in line.split(',')])
    return dat

def get_train_and_test(testmodel,traindat,trainy,testdata,testy):
    testmodel.fit(traindat,trainy)
    pres=testmodel.predict(traindat)
    print(sklearn.metrics.accuracy_score(trainy,pres))
    pres=testmodel.predict(testdat)
    testacc=sklearn.metrics.accuracy_score(testy,pres)
    print(testacc)
    return testacc

# 1 places orgs 2 people orgs 3 places people 
task='3'
project_name='retuers' # 0 20newgroups 1 
trp=1761
trn=2948-trp
tep=1172
ten=1962-tep
testfile_end='_rd30_td.txt'
if project_name=='20newgroups':
    testfile_end='_rd30_td.txt'
    if task=='2':
        trp=1197
        trn=2138-trp
        tep=796
        ten=1423-tep
    elif task=='3':
        trp=1198
        trn=2384-trp
        tep=797
        ten=1586-tep
    elif task=='4':
        trp=1188
        trn=2298-trp
        tep=790
        ten=1530-tep
    elif task=='5':
        trp=1184
        trn=2375-trp
        tep=789
        ten=1582-tep
    elif task=='6':
        trp=1181
        trn=2192-trp
        tep=786
        ten=1460-tep
elif project_name=='retuers':
    testfile_end='_td.txt'
    if task=='1':
        trp=350#6868
        trn=573-trp#7091-trp
        tep=145#2738
        ten=231-tep#2824-tep
    elif task=='2':
        trp=380
        trn=223
        tep=182
        ten=86
    elif task=='3':
        trp=350#6868
        trn=720-trp#7248-trp
        tep=145#2738
        ten=327-tep#2920-tep


testfilefolder1='C:\\Users\\I310472\\source\\repos\\doc2vec_extractor\\doc2vec_extractor\\'
#testfilefolder1='C:\\Users\\I310472\\source\\repos\\doc2vec_extractor\\doc2vec_extractor\\ep30dim30a.01\\'
#testfilefolder1='C:\\Users\\I310472\\source\\repos\\doc2vec_extractor\\doc2vec_extractor\\dim30\\'
trainfilename=task+'train'+ testfile_end
traindat=get_real_data(testfilefolder1+trainfilename)
print(len(traindat))
trainy=np.concatenate(([1]*trp,[0]*trn),axis=0)


testfilename=task+'test'+ testfile_end
testdat=get_real_data(testfilefolder1+testfilename)
print(len(testdat))
testy=np.concatenate(([1]*tep,[0]*ten),axis=0)

print('svm')
avg_svm=[]
for i in range(30):
    svmlin=sklearn.svm.LinearSVC(random_state=(i+1)*13)
    avg_svm.append(get_train_and_test(svmlin,traindat,trainy,testdat,testy))

save_svm_file=task+'test'+ project_name+'.txt'
fsvm=open(save_svm_file,'w')
fsvm.write(' '.join(str(e) for e in avg_svm))
fsvm.write('\n')
testv=np.mean(avg_svm)
fsvm.write(str(np.mean(avg_svm)))
fsvm.write('\\pm')
fsvm.write(str(np.std(avg_svm)))
fsvm.write('\n')

print('nc')
fsvm.write('\nnc\n')
ncid=NearestCentroid()
testacc=get_train_and_test(ncid,traindat,trainy,testdat,testy)
fsvm.write(str(testacc))

print('knn')
fsvm.write('\nknn\n')
neigh = KNeighborsClassifier(n_neighbors=3)
testacc=get_train_and_test(neigh,traindat,trainy,testdat,testy)
fsvm.write(str(testacc))


print('rft')
fsvm.write('\nrft\n')
avg_svm=[]
for i in range(30):
    neigh = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=(i+1)*13)
    avg_svm.append(get_train_and_test(neigh,traindat,trainy,testdat,testy))
fsvm.write(' '.join(str(e) for e in avg_svm))
fsvm.write('\n')
testv=np.mean(avg_svm)
fsvm.write(str(np.mean(avg_svm)))
fsvm.write('\\pm')
fsvm.write(str(np.std(avg_svm)))
fsvm.write('\n')

fsvm.close()