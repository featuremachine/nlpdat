

from gensim import corpora,models,similarities,utils
import os
import re
import sklearn.svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
import nltk

def save_reduced_file(strpath,dat):
    tmpf= open(strpath,"w")
    for v in dat:
        #tmpf.write(','.join(map(str, v))) #format(x, "10.3f") for x in v
        tmpf.write(','.join(str(e) for e in v))
        tmpf.write('\n')

def get_sublabls(strpath):
    with open(strpath) as fp:  
         line = fp.readline().rstrip()
         onecat=re.split('[ ]',line)
         line = fp.readline().rstrip()
         another=re.split('[ ]',line)
         return onecat,another
    return [],[]

def save_tf_file(strcontent,frd,strid,dv):
    wd=dict()
    for w in strcontent:
        if wd.get(w):
            wd[w]=wd[w]+1
        else:
            wd[w]=1
    frd.write(strid+';')
    for key,item in wd.items():
        #frd.write(str(idw[key])+','+str(item)+';')
        if dv.get(key):
            frd.write(str(dv[key])+','+str(item)+';')
    frd.write('\n')

    #getdocs(plabel,trainFolder,'TRAIN',docs,pdocs_td,pdocs_sd,ftrain_sd,ftrain_td,dv)
def getdocs(onelable,trainFolder,subfolder,docs,docs_td,docs_sd,frd_sd,frd_td,dv):
    for filefolder in onelable:
        #sd,td= get_sublabls(filefolder+'_split')
        sd,td= get_sublabls(filefolder+'_split_new.txt')
        for folderone in sd:
            testfiledir=trainFolder+"\\"+subfolder+"\\"+filefolder+'\\'+folderone+"\\"
            if os.path.exists(testfiledir)==False:
                continue
            for filePathone in os.listdir(testfiledir): 
                strv = open(testfiledir+"\\"+filePathone,'r').read()
                strv=strv.replace('\n',' ')
                strcontent=utils.simple_preprocess(strv)
                #strcontent=nltk.word_tokenize(strv)
                T=models.doc2vec.TaggedDocument(strcontent,[folderone+filePathone])
                docs.append(T)
                docs_sd.append(strcontent)
                save_tf_file(strcontent,frd_sd,filePathone,dv)
        for folderone in td:
            testfiledir=trainFolder+"\\"+subfolder+"\\"+filefolder+'\\'+folderone+"\\"
            if os.path.exists(testfiledir)==False:
                continue
            for filePathone in os.listdir(testfiledir):
                strv = open(testfiledir+"\\"+filePathone,'r').read()
                strv=strv.replace('\n',' ')
                strcontent=utils.simple_preprocess(strv)
                #strcontent=re.split('[, \n/?;.]',strv)
                #strcontent=nltk.word_tokenize(strv)
                T=models.doc2vec.TaggedDocument(strcontent,[folderone+filePathone])
                docs.append(T)
                docs_td.append(strcontent)
                save_tf_file(strcontent,frd_td,filePathone,dv)
    return docs,docs_sd,docs_td

def save_to_file(model, docs_one,strname):
    dat=[]
    for doc in docs_one:
        rslt= model.infer_vector(doc)
        dat.append(rslt)
    save_reduced_file(strname,dat)
def load_dict(strdict):
     dc=dict()
     with open(strdict) as fp:  
         line = fp.readline().rstrip()
         while line:
            vs=re.split('[, ]',line)
            if len(vs)!=2:
                line = fp.readline().rstrip()
                continue
            dc[vs[0]]=vs[1]
            line = fp.readline().rstrip()
     return dc

#trainvoc="C:\\Works\\nlp\\tasks\\ruters\\dat\\vocabulary.txt"
trainFolder="C:\\Users\\I310472\\Downloads\\one\\docs\\"
testFolder="C:\\Works\\nlp\\tasks\\tl_retuer\\dat\\"

dv=load_dict('C:\\Users\\I310472\\source\\repos\\doc2vec_extractor\\doc2vec_extractor\\new_returer\\ndict_int.txt')

docs=[]
itask='3'
bsave=True
plabel=nlabel=[]
pstrfile_save=''
nstrfile_save=''
if itask=='1':
    plabel=['places']
    nlabel=['orgs']
    pstrfile_save='places_'
    nstrfile_save='orgs_'
elif itask=='2':
    plabel=['people']
    nlabel=['orgs']
    pstrfile_save='people_'
    nstrfile_save='orgs_'
elif itask=='3':
    plabel=['places']
    nlabel=['people']
    pstrfile_save='places_'
    nstrfile_save='people_'


pdocs_td=[]
pdocs_sd=[]
#ndocs_td=[]
#ndocs_sd=[]

ftrain_sd= open(itask+pstrfile_save+"_train_sd.txt",'w')
ftrain_td= open(itask+pstrfile_save+"_train_td.txt",'w')
docs,pdocs_sd,pdocs_td=getdocs(plabel,trainFolder,'TRAIN',docs,pdocs_td,pdocs_sd,ftrain_sd,ftrain_td,dv)
ftrain_sd.close()
ftrain_td.close()

print(len(pdocs_sd))
print(len(pdocs_td))

ftrain_sd= open(itask+nstrfile_save+"_train_sd.txt",'w')
ftrain_td= open(itask+nstrfile_save+"_train_td.txt",'w')
docs,pdocs_sd,pdocs_td=getdocs(nlabel,trainFolder,'TRAIN',docs,pdocs_td,pdocs_sd,ftrain_sd,ftrain_td,dv)
ftrain_sd.close()
ftrain_td.close()

print(len(pdocs_sd))
print(len(pdocs_td))

model = models.Doc2Vec(docs,alpha=0.025,min_alpha=0.012,min_count=2,vector_size=30,workers=4,epochs=60)

#model.infer_vector(pdocs_sd)
save_to_file(model,pdocs_td,itask+'train_td.txt')
save_to_file(model,pdocs_sd,itask+'train_sd.txt')

testdocs=[]
ptest_docs_td=[]
ptest_docs_sd=[]
ftrain_sd= open(itask+pstrfile_save+"_test_sd.txt",'w')
ftrain_td= open(itask+pstrfile_save+"_test_td.txt",'w')
testdocs,ptest_docs_sd,ptest_docs_td=getdocs(plabel,trainFolder,'TEST',testdocs,ptest_docs_td,ptest_docs_sd,ftrain_sd,ftrain_td,dv)
ftrain_sd.close()
ftrain_td.close()

print(len(ptest_docs_sd))
print(len(ptest_docs_td))

ftrain_sd= open(itask+nstrfile_save+"_test_sd.txt",'w')
ftrain_td= open(itask+nstrfile_save+"_test_td.txt",'w')
testdocs,ptest_docs_sd,ptest_docs_td=getdocs(nlabel,trainFolder,'TEST',testdocs,ptest_docs_td,ptest_docs_sd,ftrain_sd,ftrain_td,dv)
ftrain_sd.close()
ftrain_td.close()

print(len(ptest_docs_sd))
print(len(ptest_docs_td))

save_to_file(model,ptest_docs_td,itask+'test_td.txt')