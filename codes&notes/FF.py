#!/usr/bin/env python
# coding: utf-8

# In[1]:
'''
Federated-Forest
2019.9.25 -2019.10.6
Greilfang
greilfang@gmail.com
'''
# In[2]:
from mpi4py import MPI
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs
import random
from collections import Counter
import math
import numpy as np
import copy
import pandas as pd
from simulation import load_random
import time

# In[3]:


'''
produce the dataset && get the dataset according to index
'''

def simulated_split_dataset(digit):
    datasets = [{} for c in range(0,hyperparams['client_num']+1)]
    all_features=[[] for c in range(0,hyperparams['client_num']+1)]
    datasets[0] = digit
    for client in range(1,hyperparams['client_num']+1):
        datasets[client]['target'] = digit['target']
        datasets[client]['data'] = None
    if(hyperparams['client_num']!=0):
        for i in range(digit['data'].shape[1]):
            client = i % hyperparams['client_num'] + 1
            #print('check none:',datasets[client]['data'])
            if datasets[client]['data'] is None:
                datasets[client]['data'] = digit['data'][:,i]
            else:
                datasets[client]['data'] = np.column_stack([datasets[client]['data'],digit['data'][:,i]])
            all_features[client].append(i)
    else:
        print("No client! No Split data!")
    return datasets,all_features

def produce_median(vals):
    medians=[]
    for i in range(1,len(vals)):
        medians.append((vals[i]+vals[i-1])/2)
    return medians

def splitIndexes(dataset,col,val,indexes):
    ind_2,ind_1=[],[]
    for i in indexes:
        if dataset['data'][i][col]>val:
            ind_2.append(i)
        else:
            ind_1.append(i)
    return ind_1,ind_2
def packMessage(gini,feature,value,rank):
    return{
        'rank':rank,
        'gini':gini,
        'feature':feature,
        'value':value
    }

def notify_selected_data(rand_rate,data_num):
    selected_num = int(rand_rate * data_num)
    index = random.sample(range(data_num),selected_num)
    return index

def notify_selected_feature(rand_rate,feature_num):
    selected_num = int(rand_rate * feature_num)
    selected_features=[[]]
    for i in range(1,hyperparams['client_num']+1):
        selected_num = int(rand_rate * feature_num / hyperparams['client_num'] )
        features=random.sample(all_features[i],selected_num)
        selected_features.append(features)
        selected_features[0].extend(features)
    
    return selected_features

def gini(data,ind):
    stats=Counter(data['target'][ind])
    all_nums=len(ind)
    result=1
    for amt in stats.values():
        result=result-(amt/all_nums)**2
    return result

def getEntropy(all_nums,class_nums):
    entropy=0
    for nums in class_nums:
        if nums!=0:
            entropy = entropy - (nums/all_nums) * math.log((nums/all_nums),2)
    return entropy

def getConditionEntropy(dataset,col,all_nums):
    fc_record={}
    for i in range(len(dataset['target'])):
        val,cls=dataset['data'][i][col],dataset['target'][i]
        if dataset['data'][i][col] in fc_record:
            fc_record[val][cls]=fc_record[val][cls]+1
        else:
            fc_record[val] = [0 for c in range(datparams['class_num'])]
            fc_record[val][cls] = 1
    condition_entropy = 0
    for val,cls in zip(fc_record.keys(),fc_record.values()):
        val_sum = sum(cls)
        condition_entropy = condition_entropy + (val_sum/all_nums) * getEntropy(val_sum, cls)
    return condition_entropy

def load_credits(matrix,rate):
    train_set,test_set={},{}
    edge = int(matrix.shape[0]*(1-rate))
    train_set['data'] = matrix[:edge,1:]
    train_set['target'] = matrix[:edge,0]
    test_set['data']=matrix[edge:,1:]
    test_set['target']=matrix[edge:,0]
    return train_set,test_set

def load_propotion_credits(matrix,good_num,bad_num,test_num):
    train_set,test_set={},{}
    rows=matrix.shape[0]
    indexes =[ i for i in range(rows)]
    test_index=random.sample(indexes,test_num)

    test_set['data']=matrix[test_index,1:]
    test_set['target']=matrix[test_index,0]

    remain_index=list(set(indexes)-set(test_index))
    
    good_index,bad_index=[],[]
    for index in remain_index:
        if matrix[index,0] == 1:
            good_index.append(index)
        else:
            bad_index.append(index)

    good_sample=random.sample(good_index,good_num)
    bad_sample=random.sample(bad_index,bad_num)
    samples = good_sample+bad_sample
    random.shuffle(samples)

    train_set['data'] = matrix[samples,1:]
    train_set['target'] = matrix[samples,0]
    
    return train_set,test_set

def load_equal_credits(matrix,good_num,bad_num,test_ratio):
    train_set,test_set={},{}
    rows=matrix.shape[0]
    indexes =[ i for i in range(rows)]
    good_index,bad_index=[],[]
    for index in indexes:
        if matrix[index,0] == 1:
            good_index.append(index)
        else:
            bad_index.append(index)
    
    good_train=random.sample(good_index,good_num)
    bad_train=random.sample(bad_index,bad_num)

    rest_good_sample=list(set(good_index)-set(good_train))
    rest_bad_sample=list(set(bad_index)-set(bad_train))

    good_test=random.sample(rest_good_sample,int(good_num*test_ratio))
    bad_test=random.sample(rest_bad_sample,int(bad_num*test_ratio))
    
    trains = good_train+bad_train
    tests = good_test+bad_test
    random.shuffle(trains)
    random.shuffle(tests)

    train_set['data'] = matrix[trains,1:]
    train_set['target'] = matrix[trains,0]
    test_set['data'] = matrix[tests,1:]
    test_set['target'] = matrix[tests,0]
    
    return train_set,test_set
class FederatedDecisionTreeClassifier:
    def __init__(self,dataset,rank,client_num):
        # symbolize comm.rank()
        self.rank = rank
        self.client_num = client_num
        # dataset is the actual dataset a client owns
        self.dataset = dataset
        # classifier structure
        self.structure = None
        # prevent the repetition in split threshold
        self.threshold_map = {}
        self.tests = None
        self.leaves=[]
        
    
    def need_pruning(self,indexes):
        if len(indexes)<4 :
            return True
        return False

    def bagging(self,indexes):
        #print('rank： ',self.rank,' ',indexes)
        stats=Counter(self.dataset['target'][indexes])
        if stats[0] >=stats[1]:
            return 0
        else:
            return 1
        #prediction = max(stats,key = stats.get)
        #return prediction

    def calculateBestInfoGain(self,dataset,features):
        all_nums = len(dataset['target'])
        class_nums = list(Counter(dataset['target']).values())
        best_info_gain,best_split_feature = float('-Inf'),None
        entropy = getEntropy(all_nums,class_nums)
        
        for col in range(dataset['data'].shape[1]):
            condition_entropy = getConditionEntropy(dataset,col,all_nums)
            info_gain = condition_entropy-entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_feature = features[col]
        return best_info_gain,best_split_feature

    def repeated(self,features,i,val):
        if features[i] in self.threshold_map and val in self.threshold_map[features[i]]:
            return True
        return False
        
    def calculateBestGini(self,indexes,features):
        rows,cols=len(indexes),len(features)
        best_gini_gain,split_feature,split_value=float('Inf'),-1,None
        #print('indexes:\n',indexes)
        for i in range(cols):
            medians=produce_median(list(set(self.dataset['data'][indexes,i])))
            #print('medians:\n',medians)
            for val in medians:
                ind_1,ind_2=splitIndexes(self.dataset,i,val,indexes)
                p = len(ind_1) / rows
                gini_gain = p*gini(self.dataset,ind_1)+(1-p)*gini(self.dataset,ind_2)
                if gini_gain < best_gini_gain and not self.repeated(features,i,val):
                    best_gini_gain,split_feature,split_value=gini_gain,features[i],val
        
        return best_gini_gain,split_feature,split_value
            
        
    def getBestClient(self,message):

        reply = {'sign':'Success','feature':None,'threshold':float('inf')}
        dest = 0
        for i in range(1,len(message)):
            if message[i]['feature']!=-1 and message[i]['value'] < reply['threshold']:
                reply['threshold'] = message[i]['value']
                reply['feature'] = message[i]['feature']
                dest = message[i]['rank']
        
        if reply['feature']==-1:
            reply['sign']='Prune'
        return dest,reply
    
    def buildTree(self,indexes,features):
        self.structure=self.FederatedTreeBuild(indexes,features)
    
    #dataset will change through the split of tree
    def FederatedTreeBuild(self,indexes,features):
        node = {'feature':None,'threshold':None,'gini':None,'left':None,'right':None}
        if  len(set(self.dataset['target'][indexes]))==1 :
            ind=indexes[0]
            node['class']=self.dataset['target'][ind]
            return node
        
        best_gini_gain,split_feature,split_value = None,None,None
        message=[]
        if len(features)!=0 and self.rank!=0:
            best_gini_gain,split_feature,split_value = self.calculateBestGini(indexes,features)
            message = packMessage(best_gini_gain,split_feature,split_value,self.rank)
            node['feature'],node['threshold'],node['gini']=split_feature,split_value,best_gini_gain
        
        message = comm.gather(message,root = 0)

        success_reply = None
        
        left_indexes,right_indexes=None,None
        is_selected = False
        prune_notice=None
        if self.rank == 0:
            print('message\n',message)
            destination,success_reply = self.getBestClient(message)
            if success_reply['sign']=='Prune':
                prune_notice=success_reply
        
        prune_notice = comm.bcast(prune_notice if self.rank ==0 else None,root=0)
        if prune_notice is not None:
            node['class']=self.bagging(indexes)
            return node


        if self.rank==0:
            req = comm.isend(success_reply,dest = destination,tag = 1)
            req.wait()
            
            split_notice = comm.irecv(source=destination,tag = 2 )
            division_message = split_notice.wait()
            
            left_indexes = division_message['left_indexes']
            right_indexes = division_message['right_indexes']
            
            client_division = {
                'sign':'Failure',
                'left_indexes':left_indexes,
                'right_indexes':right_indexes
            }
            
            for i in range(1,self.client_num+1):
                if i != destination:
                    req = comm.isend(client_division,dest = i, tag =1)
                    req.wait()
            #print('master task over!')
        
        elif self.rank != 0:
            req = comm.irecv(source = 0,tag = 1)
            acknow = req.wait()
            if acknow['sign']=='Success':
                is_selected = True
                column = features.index(acknow['feature'])
                left_indexes,right_indexes = splitIndexes(self.dataset,column,acknow['threshold'],indexes)
                        #reply = {'sign':'Success','feature':None,'threshold':float('inf')}
                if acknow['feature'] not in self.threshold_map:
                    self.threshold_map[acknow['feature']]=[]    
                self.threshold_map[acknow['feature']].append(acknow['threshold'])

                #print('rank ',self.rank)
                #print('acknow',acknow)
                #print('left:\n',left_indexes)
                #print('right:\n',right_indexes)
                division_message={
                    'sign':'division',
                    'left_indexes':left_indexes,
                    'right_indexes':right_indexes
                }
                split_notice = comm.isend(division_message,dest = 0,tag = 2)
                split_notice.wait()
            elif acknow['sign']=='Failure':
                left_indexes,right_indexes = acknow['left_indexes'],acknow['right_indexes']
        
        
        if not is_selected:
            node['feature'],node['threshold'],node['gini']=None,None,None            
        
        node['left'] = self.FederatedTreeBuild(left_indexes,features)
        node['right']= self.FederatedTreeBuild(right_indexes,features)
        return node
                  
    def predict(self,dataset):
        self.tests = dataset
        indexes =[i for i in range(dataset['data'].shape[0])]
        self.FederatedTreePredict(indexes,self.structure)

    def FederatedTreePredict(self,indexes,node):
        if self.rank != 0:
            if 'class' in node:
                self.leaves.append({'indexes':indexes,'prediction':node['class']})
            elif node['feature'] is None:
                self.FederatedTreePredict(copy.deepcopy(indexes),node['left'])
                self.FederatedTreePredict(copy.deepcopy(indexes),node['right'])
            elif node['feature'] is not None:
                left_ind,right_ind = splitIndexes(self.tests,node['feature'],node['threshold'],indexes)
                self.FederatedTreePredict(left_ind,node['left'])
                self.FederatedTreePredict(right_ind,node['right'])

                
class FederatedForestClassifier:
    def __init__(self,n_tree,rank,rate,client):
        self.forest = []
        self.tree_num = n_tree
        self.rank = rank
        self.rate = rate
        
        self.voters=None
        self.dataset = None
        self.tests=None
        if rank == 0:
            self.client_num = client
        else:
            self.client_num = None
        
    def fit(self,dataset):
        self.dataset = dataset
        data_num = self.dataset['data'].shape[0]
        feature_num = self.dataset['data'].shape[1]
        #self.voters = np.zeros((data_num,self.tree_num))
        
        for i in range(self.tree_num):
            if self.rank==0:
                print('Tree ',i)
                print('--------------------------------------------------------------------------')
            selected_index=[]
            if self.rank == 0:
                selected_index = notify_selected_data(self.rate,data_num)
                selected_features = notify_selected_feature(1,feature_num)
            elif self.rank != 0:
                selected_index = None
                selected_features = None
            
            selected_index = comm.bcast(selected_index if self.rank == 0 else None,root = 0)
            selected_features = comm.scatter(selected_features if self.rank == 0 else None, root = 0)
            fdtc=FederatedDecisionTreeClassifier(copy.deepcopy(self.dataset),self.rank,self.client_num)
            
            fdtc.buildTree(selected_index,selected_features)
            self.forest.append(fdtc)
            
    def predict(self,dataset):
        self.tests=dataset
        data_num = self.tests['data'].shape[0]
        self.voters = np.zeros((data_num,self.tree_num))
        for i in range(self.tree_num):
            classify_result=[]
            if self.rank != 0:
                self.forest[i].predict(dataset)
                classify_result = self.forest[i].leaves
            
            classify_result = comm.gather(classify_result,root = 0)
            
            if self.rank == 0:
                real_leafs = self.getUnion(classify_result)
                self.generateTarget(i,real_leafs)
        
        if self.rank == 0:
            self.bagging()
                
    def getUnion(self,result):
        real_leafs=[]
        leaf_num = len(result[1])
        client_num = self.client_num
        test_num = self.tests['data'].shape[0]
        
        for i in range(leaf_num):
            inter=set(list(x for x in range(test_num)))
            for j in range(1,client_num+1):
                inter = inter.intersection(set(result[j][i]['indexes']))
            #print('intersection ',i,':\n',inter)
            real_leafs.append({'indexes':list(inter),'prediction':result[1][i]['prediction']})
        return real_leafs
    
    
    def generateTarget(self,tree,real_leafs):
        for real_leaf in real_leafs:
            self.voters[real_leaf['indexes'],tree] = real_leaf['prediction']
            
    def bagging(self):
        test_num = self.tests['data'].shape[0]
        self.prediction = [-1 for x in range(test_num)]
        for i in range(test_num):
            self.prediction[i] = max(self.voters[i,:],key=list(self.voters[i]).count)
        #print('final bagging\n',self.prediction)
    def getAccuracy(self,target):
        all_predict,true_predict=0,0
        class_0,class_1=0,0
        acc_0,acc_1=0,0
        assert(len(self.prediction)==len(target))
        for i in range(len(self.prediction)):
            if target[i]==0:
                class_0=class_0+1
            elif target[i]==1:
                class_1=class_1+1
            
            if self.prediction[i]==target[i]:
                if self.prediction[i]==0:
                   acc_0=acc_0+1
                elif self.prediction[i]==1:
                    acc_1=acc_1+1
                true_predict = true_predict+1
            all_predict = all_predict +1
        print('accuracy:',true_predict/all_predict)
        print('0 accuracy:',acc_0/class_0)
        print('1 accuracy:',acc_1/class_1)

# We represent 0 as the default server-rank
# self_rank means the rank of server-node
# comm_size means the rank of client-node
comm=MPI.COMM_WORLD
self_rank = comm.Get_rank()
comm_size = comm.Get_size()

# if self_rank == 0:
#     train_set,test_set= load_random(n_samples=800, centers=2, n_features=10,test_rate=0.2)

# train_set = comm.bcast(train_set if self_rank == 0 else None,root = 0)
# test_set = comm.bcast(test_set if self_rank == 0 else None,root = 0)
# test_set=copy.deepcopy(train_set)
hyperparams={
    'client_num':comm_size - 1,
    'tree_num':40,
    'rand_data_rate':0.025,
    'rand_feature_rate':1
}
datparams={
    'data_num':32000,
    'test_num':1200,
    'feature_num':10,
    'class_num':2
}

train_set,test_set=None,None
if self_rank==0:
    csv_data = pd.read_csv('CleanedData.csv')
    #train_set,test_set=load_credits(csv_data.values[:datparams['data_num']],rate=0.2)
    #train_set,test_set=load_propotion_credits(csv_data.values,37500,2500,datparams['test_num'])
    train_set,test_set=load_equal_credits(csv_data.values,28800,3200,datparams['test_num']/datparams['data_num'])

train_sets=None
if self_rank ==0:
    train_sets,all_features = simulated_split_dataset(train_set)

train_set = comm.scatter(train_sets, root=0)
test_set = comm.bcast(test_set if self_rank == 0 else None,root = 0)

timw_start,time_end=None,None
if self_rank==0:
    time_start = time.time()
ffc=FederatedForestClassifier(
    n_tree = hyperparams['tree_num'],
    rank = self_rank,
    rate = hyperparams['rand_data_rate'],
    client=hyperparams['client_num']
)
ffc.fit(copy.deepcopy(train_set))
ffc.predict(test_set)
if self_rank ==0:
    time_end=time.time()
    ffc.getAccuracy(test_set['target'])
    print('total time:', time_end-time_start)


#%%
