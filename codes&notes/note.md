## Federated Forest
先说明$ joint\ model$是有必要的
$\to$ 如何训练一个 $joint\ model$
$\to$ 针对$federated\ learning$的概念,提出一个具体的模型$federated\ forest$

### Introduction
**四大贡献**
+ Secured privacy
>每个节点对其他节点是透明的,信息交换尽可能小
+ Loseless
>适用于$vertical\ federated\ learning$,相比于集中式 $federated\ learning$ 可以做到无损
+ Efficiency
> 使用了MPI方法即时交换信息
> 简单不受限的预测算法(树的节点数,分裂数等)
+ 可行性和广泛性
> 支持分类和回归
> 易于实现,一样准确,有效率和鲁棒

### Related Work
其他论文的相关工作,引入模型,应用联邦学习,以及联邦学习的分类
#### Federated learning
1. meta learning
2. multi-task
3. AdaBoost
4. logistic regression
5. vertical, horizontal, transfer 三种分类
6. secured federated transfer learning
7. 增强学习还有基于树的
#### Data Privacy Protection
**differential privacy 差分隐私**
+ 不增加计算负担
+ 降低模型表现

**homomorphic encryption 同态加密**
+ 算法复杂
+ 不支持$none\ linear\ model$(可以用泰勒展开近似)

### Problem Formulation
$The\ model\ is\ assumed\ a\ vertical\ federated\ learning.$

$There\ are\ M\ nodes $
$To\ each\ node:$
$$The\ data\ dormain\ is\ D_i$$
$To\ all\ models:$
$$ Tha\ data\ dormain\ is\ D=D_1 \small \bigcup D_2 \small \bigcup ...  \small \bigcup D_m$$ $$where\ 1\leq i \leq M$$

$Denote\ the\ feature\ space\ of\ D_m\ as\ F_m:$
$$As\ above:F=F_1 \small \bigcup F_2 \small \bigcup ...  \small \bigcup F_m$$ $$where\ 1\leq i \leq M$$

$\text{All features' true name are encoded in order to protect privacy}$

$As\ assmuption,\ for\ 1\leq i,j\leq M:$
$$if\ i \neq j\ then\ F_i \small \bigcap F_j=\emptyset$$
$\text{In the work, sample nums are same and IDs are in accord}$
  
**Notation**
+ 现实中$M$通常很小
+ 不讨论$ID$如何对齐这种问题

基本架构和 $federated\ learning$ 基本一样,$label\ y$ 由1个节点提供

### Problem Statement

**Given:** Regional domain $D_i$ and encrypted label $y$ on each client $i, 1 ≤ i ≤ M. $
**Learn:** A Federated Forest, such that for each tree in the forest: 
1) a complete tree model $T$ is held on master; 
2) a partial tree model $T_i$ is stored on each client $i, 1 ≤ i ≤ M.$ 

**Constraint:** The performance (accuracy, f1-score, MSE, e.t.c.) of the Federated Forest must be comparable to the non-federated random forest.
### Methodology

对于单个节点有$Alogorithm1$

$Alogorithm\ 1:$

```Python
while 还有树要建立:
    if (node收到 Fi,Di):
    Function TreeBuild(){
        建立一个空的树节点
        if 满足剪枝条件 :
            设置为叶节点,voting得出label
        
        '''设置要记录的初始量'''
        purify,f_plus = -∞ , None
        '''
        如果master随机取到的fetures在该节点内还有没被作为分割点的
        注意这里是计算,并不划分
        '''
        if Fi != None:
            计算每个节点做分割点的信息增益,加入impurity_improvements
            '''得到最好的,f_plus记录该最好的节点'''
            best_purity = MAX(impurity_improvements)
            f_plus=getNodeOf(best_purity)
        
        把 best_purity发给master

        '''master告诉该节点这是全局最优'''
        if 从master接收到的split_message:
            '''节点正式划分'''
            is_selected=True
            划分样本
            把 left_tree, right_tree发给master
        else:
        '''说明最优划分节点不是这课树'''
            收到left_tree,right_tree各有哪些节点
        
        '''递归向下建树'''
        left_tree=TreeBuild(left_tree)
        right_tree=TreeBuild(right_tree)
        return node
    '''
    一棵树建立完了,加入森林
    可以把这个算法理解为,将计算信息增益的任务平摊到了每个节点
    '''
    Forests.append(Tree)
}
```

针对master节点
$Alogorithm\ 2:$
```Python
while 还需要建树 :
    随机取D,F
    对每个节点,把该节点的Fi发给该节点
    Fuction TreeBuild(D,F,y):
        if 满足剪枝条件:
            设置为叶节点,voting得出label
        
        接受来自各个节点的增益值,node_best_purities
        global_best_purity=MAX(node_best_purities)
        '''确定最优划分特征的来源节点'''
        selected_node=getNodeOf(global_best_purity)

        for node in nodes:
            if node is selected_node:
                发split_messagte表示它被选中
            else:
                把划分结果left_tree,right_tree告知node
        
    返回node根节点
    Forest.append(Tree)

```

单个client无法包含一个划分节点的特征的全部信息,但每个client的森林结构是一样的.


针对client节点的预测
$Alogorithm\ 3:$
```Python
while 需要预测:
    TreePrediction(Ti,Di_test,Fi):
        '''Si是要预测的一个样本,in leaf表示预测完成'''
        if Si is in leaf:
            return (Si,Labeli)
        else:
            if 节点保存了划分信息(最优节点) :
                划分
                left_tree=TreePrediction(Ti_left,Di_test_left,Fi)
                right_tree=TreePrediction(Ti_right,Di_test_right,Fi)
            else:
                left_tree=TreePrediction(Ti_left,Di_test_left,Fi)
                right_tree=TreePrediction(Ti_right,Di_test_right,Fi)
            return (left_tree,right_tree)
        
        把 (S1,S2,...,Sm) 送到master
```
根据论文,如果一个样本遇到有划分标准的节点,则划分,否则同时全部进入左右两个节点

对于决策树 $Ti$ 的每个叶节点,都会有一批样本,对第l个叶节点里的样本,记作 $S_i^l$, $(l \in L$, $L$是$T_i$的叶节点集合)

对 $\{S_i^l\}_{i=1}^M$ 取交集运算

每个节点的森林的结构是一样的,因此所有叶节点的位置和数目一一对应,可以做交集运算

虽然论文 appendix 的证明正规冗长,理解起来是不难的:
可以理解为:
$$S\small\bigcap A\small\bigcap B\small\bigcap C = (S\small\bigcap A\small\bigcap B)\small\bigcap (S\small\bigcap C)$$
$$\text{S是样本全集. A,B,C是分别满足条件a,b,c的样本集合}$$

$Algorithm\ 4:$
```Python
while 需要预测:
    Gather{S1,S2,...Sm}
    Obtain(S^1,S^2,...S^m)
Return all the label 
```

### Privacy Protection
**Identities:** 对用户进行hash编码然后进行MD5加密
**Label:** encode和同态加密的权衡
**Feature:** feature名进行加密
**Communication:** RSA或ADS
**Moddel Storage:**  $Federated\ Learning$ 的细节已经实现对模型的保护

### Experiment
重点不是表现多好

而是$Federated\ Learning$ 和 $Classical\ Random\ Forest$ 表现的比较

### 结论
$Federated\ Forest$ 相比于 $Classical\ Random\ Forest$ 速度快且无损.





