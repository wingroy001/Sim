# -*- coding: utf-8 -*-
from __future__ import division 
import networkx as nx
from copy import deepcopy
import random,json,time
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.metrics import auc
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
#matplotlib.use("Agg")
plt.switch_backend('agg')
from sklearn.metrics import precision_recall_curve
from xlutils.copy import copy
import xlwt, xlrd
from scipy import interpolate 
t1=time.time()
graphname=['hi-iii','hi-ii-14','hi-tested','lit-bm-13','lit-nb-13','Arabidopsis_thaliana2','Arabidopsis1','BIOGRID-ORGANISM-Caenorhabditis_elegans','BIOGRID-ORGANISM-Human_Immunodeficiency_Virus','BIOGRID-ORGANISM-Rattus_norvegicus-3.5.171.tab','Candida_albicans','coli1','Drosophila_melanogaster1','elegans','Escherichia_coli1','Hamster','Homo_sapiens2','Human_Herpesvirus','Human1','marina1','mouse1','MovieRate2','Mus_musculus2','Oryza1','Plasmodium_falciparum','Schizosaccharomyces1','USAir','Xenopus','Yang','yeast1','YeastS','Human33']

###############################################################################
#从这里看开始替换脚本中的网络名称


print 'Network: Arabidopsis_thaliana2'
G0=nx.read_weighted_edgelist('//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2.txt') #读取原图
print 'nx.info(G0)'
print nx.info(G0)
print 'average clustering coefficient:', nx.average_clustering(G0)
nx.write_weighted_edgelist(G0, '//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_00.txt')
G0=nx.read_weighted_edgelist('//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_00.txt') #重新读取原图    
G1=nx.empty_graph(G0.nodes()) #初始化第一轮的正测试集为空图
samplesize=int(0.1*len(G0.edges))  #正负测试集总规模=samplesize*2
iteration=10
print 'iteration=',iteration
for i in range(iteration):  
  edges1 = random.sample(G0.edges(), samplesize) #原图中随机取10%条边    
  G0.remove_edges_from(edges1) #删除这一轮选择的10%的边
  G0.add_edges_from(G1.edges()) #添加上一轮删除的10%的边，保证G0拥有90%的边
  nx.write_weighted_edgelist(G0, '//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_01_%d.txt'%(i+1)) #训练集
  G1=nx.empty_graph(G0.nodes())  #初始化这一轮的正测试集为空图
  G1.add_edges_from(edges1)  
  nx.write_weighted_edgelist(G1, '//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_02_%d.txt'%(i+1))#正样本 
  print 'positive samplesize:', len(G1.edges)
  print 'training samplesize:', len(G0.edges)
print 'sample phase is done'
t2=time.time()
print("time:%s"%(t2-t1))  
print 'L3 processing...'

###############################################################################
yy1,yy2,yy3,yy4=[],[],[],[]
Pre,Rec=[],[]
for kfold in range(iteration):  
  G0=nx.read_weighted_edgelist('//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_00.txt', delimiter=' ', 
                             create_using=None, nodetype=None, encoding='utf-8')
  G= nx.read_weighted_edgelist('//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_01_%d.txt'%(kfold+1), delimiter=' ', 
                             create_using=None, nodetype=None, encoding='utf-8')#训练集
  G1= nx.read_weighted_edgelist('//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_02_%d.txt'%(kfold+1), delimiter=' ', 
                             create_using=None, nodetype=None, encoding='utf-8')#正样本 
  G2= nx.complement(G)
  edges=[i for i in G1.edges] #取所有正样本
  G2.remove_edges_from(edges)  #在训练图的补图中删去正样本，做为负样本
  #G0原始图，G训练集，G1正样本，G2负样本
  n=len(G0.nodes()) #原始图节点数
  m=len(G0.edges)   #原始图边数
  G.add_nodes_from(G0.nodes()) 
  G1.add_nodes_from(G0.nodes())
  G2.add_nodes_from(G0.nodes()) 
############################################################################### 
  labels1,scores1,scores3 = [],[],[]
  for (i, j, d) in G2.edges(data=True): 
    aa,bb=0,0
    for v in G.neighbors(j):
        a,b=0,0
        for u in G.neighbors(i): 
            if (u,v) in G.edges:
                a+=1/(G.degree(u)*G.degree(v))
                b+=1/(G.degree(u)**0.5*G.degree(v)**0.5)
        aa+=a
        bb+=b
    labels1.append(1)  #负样本标签是1
    scores3.append(aa) #L3+负样本数值
    scores1.append(bb) #L3负样本数值
############################################################################### 
  for (i, j, d) in G1.edges(data=True): 
    aa,bb=0,0
    for v in G.neighbors(j):
        a,b=0,0
        for u in G.neighbors(i): 
            if (u,v) in G.edges:
                a+=1/(G.degree(u)*G.degree(v))
                b+=1/(G.degree(u)**0.5*G.degree(v)**0.5)
        aa+=a 
        bb+=b
    labels1.append(2)  #正样本标签是2
    scores3.append(aa) #L3+正样本数值   
    scores1.append(bb) #L3正样本数值
###############################################################################    
  print 'L3 is done'
  t3=time.time()
  print("time:%s"%(t3-t2))
  print 'Sim processing...'
###############################################################################
  #Sim
  def jacc2(u, v):  # 定义相似度为：共同邻居数在总邻居数中的比例，相似度为1表示两个向量一模一样 
    w=[u[i]+v[i] for i in range(n)] #u,v向量的维度n需要先定义，w为u和v对应分量之和
    j=n-w.count(0) #u和v的总共邻居数
    if j!=0:
       jac=w.count(2)/j 
    else:  #两个向量都是0向量
       jac=1  
    return jac 
###############################################################################
  index=[i for i in G.nodes()] #复制G的节点集为列表格式，和原节点集中节点顺序一致   

  X=np.zeros((n,n)) #初始化邻接矩阵

  for i in range(n):
    for j in range(i+1,n):
        if (index[i],index[j]) in G.edges:
            X[i,j],X[j,i]=1,1
  #print 'X='
  #print X       #邻接矩阵   
  #print jacc2(X[0], X[1])
  Y=np.zeros((n,n))  #杰卡德相似度矩阵初始化为零矩阵
  Z=np.zeros((n,n))  #链接概率矩阵初始化为零矩阵
###############################################################################
  #jaccard=[]
  for i in range(n):
    for j in range(i,n):
        J=jacc2(X[i], X[j]) 
        Y[j,i],Y[i,j]=J,J
  Z=np.dot(X,Y)+np.dot(Y,X)   
  #print Y  #杰卡德相似度矩阵
  #print Z  #链接概率矩阵   
  #jaccard.sort(reverse=True)   #得分按从大到小排序输出 
  #np.savetxt("//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_Sim_jaccard.txt",jaccard,fmt='%s')
  order,k=dict(),0     #order字典格式，存储节点的秩（在节点集中的顺序）     
  for i in G.nodes():
    order[i]=k
    k+=1  
   
############################################################################### 
  scores2,simscore= [],[]
  for (i, j, w) in G2.edges(data=True):
    scores2.append(Z[order[i],order[j]]) #负样本得分
    #simscore.append((Z[order[i],order[j]],i,j,1))

  for (i, j, w) in G1.edges(data=True):
    scores2.append(Z[order[i],order[j]]) #正样本得分
    #simscore.append((Z[order[i],order[j]],i,j,2) )
    
  #simscore.sort(reverse=True)   #得分按从大到小排序输出
  #print simscore
###############################################################################
  #np.savetxt("//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_Sim_sortscore.txt", simscore, fmt='%s')
  #nx.draw(G, with_labels=True)  
  print 'Sim is done' 

  t4=time.time()
  print("time:%s"%(t4-t3))
  print 'LO processing...'
###############################################################################
  #LO
  I = np.eye(n) #单位矩阵
  alpha=0.00001  
  f=X-np.dot(X,np.linalg.inv(I+alpha*np.dot(X,X)))  #X是邻接矩阵，f是链接概率矩阵
  scores4=[]
  for (i, j, w) in G2.edges(data=True):
    scores4.append(f[order[i],order[j]]) #负样本得分
  for (i, j, w) in G1.edges(data=True):
    scores4.append(f[order[i],order[j]]) #正样本得分
###############################################################################
  t5=time.time()
  print 'LO is done'
  print("time:%s"%(t5-t4))
  print 'Top k precision processing...'
###############################################################################
  #top k  
  x=zip(scores1,labels1) #将score和label打包  L3
  x.sort(reverse=True) #按score从大到小排序，label跟着排序
  y=zip(scores2,labels1) #将score和label打包 Sim
  y.sort(reverse=True) #按score从大到小排序，label跟着排序
  x1=zip(scores3,labels1) #将score和label打包  L3Plus
  x1.sort(reverse=True) #按score从大到小排序，label跟着排序
  z=zip(scores4,labels1) #将score和label打包  LO
  z.sort(reverse=True) #按score从大到小排序，label跟着排序
  w1,w2,w3,w4=[],[],[],[] #初始化precision
  pos1,pos2,pos3,pos4,k=0,0,0,0,min(1000,len(labels1)) #前k次预测中正样本的个数
  for i in range(k):
    pos1+=int(x[i][1]/2)  #正样本标签是2，负样本标签是1，标签处以2取整之和即为正样本个数
    pos2+=int(y[i][1]/2)
    pos3+=int(x1[i][1]/2) 
    pos4+=int(z[i][1]/2) 
    w1.append(pos1/(i+1)) 
    w2.append(pos2/(i+1)) 
    w3.append(pos3/(i+1)) 
    w4.append(pos4/(i+1)) 

###############################################################################
  if k<100:
    K=[k]
  elif k>=100 and k<500:
    K=[100]
  elif k>=500 and k<1000:
    K=[100,500]
  elif k>=1000:
    K=[100,500,1000]

  for k in K:
    x=[i for i in range(1,k+1)]
    y1=[w1[i] for i in range(k)]
    y2=[w2[i] for i in range(k)]
    y3=[w3[i] for i in range(k)]
    y4=[w4[i] for i in range(k)]
    AUP1 = auc(x, y1)
    AUP2 = auc(x, y2)
    AUP3 = auc(x, y3)
    AUP4 = auc(x, y4)
    plt.figure(k) #图的编号
    plt.figure(figsize=(8,6)) #画布大小
    plt.title("Arabidopsis_thaliana2" ) #图顶部标签
    plt.xlabel("Top %d prediction"%k) #横坐标标签
    plt.ylabel("Precision")  #纵坐标标签
    plt.ylim(0,1) #纵坐标范围
    plt.xlim(0,k) #横坐标范围
    plt.plot(x, y1,'g',lw=1,label='AUC of L3=%0.6f'% AUP1)# 颜色，粗细，标签
    plt.plot(x, y3,'b',lw=1,label='AUC of L3plus=%0.6f'% AUP3)# 颜色，粗细，标签
    plt.plot(x, y2,'r',lw=1,label='AUC of Sim=%0.6f'% AUP2)# 颜色，粗细，标签
    plt.plot(x, y4,'y',lw=1,label='AUC of LO=%0.6f'% AUP4)# 颜色，粗细，标签
    plt.legend(loc=0) #右下角标注曲线信息
    plt.savefig("//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_%d_Top%d_Precision.jpg"%(kfold+1,k)) #保存ROC曲线图
    plt.show()# show the plot on the screen  
  yy1.append(y1) #保存方法L3前1000次预测的precision
  yy2.append(y2) #保存方法Sim前1000次预测的precision
  yy3.append(y3) #保存方法L3+前1000次预测的precision
  yy4.append(y4) #保存方法LO前1000次预测的precision
  print 'Top k precision is done'
  t6=time.time()
  print("time:%s"%(t6-t5))
  print 'Precision recall curve processing...'
###########################################################################################
  #Precision recall curve
  precision1, recall1, thresholds1 = precision_recall_curve(labels1, scores1,pos_label=2)
  precision2, recall2, thresholds2 = precision_recall_curve(labels1, scores2,pos_label=2)
  precision3, recall3, thresholds3 = precision_recall_curve(labels1, scores3,pos_label=2)
  precision4, recall4, thresholds4 = precision_recall_curve(labels1, scores4,pos_label=2)
  print 'labels1,scores1,scores2,scores3,scores4',len(labels1),len(scores1),len(scores2),len(scores3),len(scores4)
  #print 'precision1, recall1',len(precision1), len(recall1)
  #print 'precision2, recall2',len(precision2), len(recall1)
  #print 'precision3, recall3',len(precision3), len(recall3)
  #print 'precision4, recall4',len(precision4), len(recall4)

  Pre.append([precision1,precision2,precision3,precision4])
  Rec.append([recall1,recall2,recall3,recall4]) 
  AUP1 = auc(recall1, precision1)
  AUP2 = auc(recall2, precision2)
  AUP3 = auc(recall3, precision3)
  AUP4 = auc(recall4, precision4)
  plt.figure(2)
  plt.figure(figsize=(8,6)) #画布大小
  plt.title("Arabidopsis_thaliana2" ) #图顶部标签
  plt.xlabel('Recall')# make axis labels
  plt.ylabel('Precision')
  plt.ylim(0,1) #纵坐标范围
  plt.xlim(0,1) #横坐标范围
  plt.plot(recall1, precision1,'g',lw=1,label='AUC of L3=%0.6f'% AUP1)
  plt.plot(recall3, precision3,'b',lw=1,label='AUC of L3plus=%0.6f'% AUP3)
  plt.plot(recall2, precision2,'r',lw=1,label='AUC of Sim=%0.6f'% AUP2)
  plt.plot(recall4, precision4,'y',lw=1,label='AUC of LO=%0.6f'% AUP4)
  plt.legend(loc=0) #右下角标注曲线信息
  plt.savefig("//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_%d_PR_mycode3.jpg"%(kfold+1))
  plt.show()
  print 'Precision recall cure is done'
  t7=time.time()
  print("time:%s"%(t7-t6))

###############################################################################
w1,w2,w3,w4=[0]*1000,[0]*1000,[0]*1000,[0]*1000 #初始化前1000次预测的平均precision，能够预测次数的必须大于或等于1000次，否则报错！
for i in range(1000):
    for j in range(iteration):
        w1[i]+=yy1[j][i]
        w2[i]+=yy2[j][i]
        w3[i]+=yy3[j][i]
        w4[i]+=yy4[j][i]
    w1[i]=w1[i]/iteration
    w2[i]=w2[i]/iteration
    w3[i]=w3[i]/iteration
    w4[i]=w4[i]/iteration
K=[100,500,1000]
for k in K:
    x=[i for i in range(1,k+1)]
    y1=[w1[i] for i in range(k)]
    y2=[w2[i] for i in range(k)]
    y3=[w3[i] for i in range(k)]
    y4=[w4[i] for i in range(k)]
    AUP1 = auc(x, y1)
    AUP2 = auc(x, y2)
    AUP3 = auc(x, y3)
    AUP4 = auc(x, y4)
    plt.figure(k+1) #图的编号
    plt.figure(figsize=(8,6)) #画布大小
    plt.title("Arabidopsis_thaliana2" ) #图顶部标签
    plt.xlabel("Top %d prediction"%k) #横坐标标签
    plt.ylabel("Precision")  #纵坐标标签
    plt.ylim(0,1) #纵坐标范围
    plt.xlim(0,k) #横坐标范围
    plt.plot(x, y1,'g',lw=1,label='AUC of L3=%0.6f'% AUP1)# 颜色，粗细，标签
    plt.plot(x, y3,'b',lw=1,label='AUC of L3plus=%0.6f'% AUP3)# 颜色，粗细，标签
    plt.plot(x, y2,'r',lw=1,label='AUC of Sim=%0.6f'% AUP2)# 颜色，粗细，标签
    plt.plot(x, y4,'y',lw=1,label='AUC of LO=%0.6f'% AUP4)# 颜色，粗细，标签
    plt.legend(loc=0) #右下角标注曲线信息
    plt.savefig("//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_Top%d_Precision.jpg"%(k)) # 保存平均ROC曲线图
    plt.show()# show the plot on the screen 
 
##################################################################################################
new_length=500 #重采样的点数
new_x = np.linspace(0, 1, new_length) #新的横坐标new_length个点，范围是0到1，均匀采点
new_y=[0]*new_length #初始化平均曲线的纵坐标
y_all,AUP_all=[],[] #初始化4种方法的平均PR曲线纵坐标和平均AUC值
for j in range(4):  #4种方法按顺序为L3,Sim，L3+，LO
  for i in range(iteration):
    tem_y = interpolate.interp1d(Rec[i][j], Pre[i][j])  
    new_y=[new_y[i]+tem_y(new_x)[i] for i in range(new_length)] #多条曲线对应纵坐标叠加
  new_y=[new_y[i]/iteration for i in range(new_length)] #多条曲线对应纵坐标的平均值为平均曲线的纵坐标
  AUP_all.append(auc(new_x, new_y))
  y_all.append(new_y)
plt.figure(11)
plt.figure(figsize=(8,6)) #画布大小
plt.title("Arabidopsis_thaliana2" ) #图顶部标签
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')
plt.ylim(0,1) #纵坐标范围
plt.xlim(0,1) #横坐标范围
plt.plot(new_x, y_all[0],'g',lw=1,label='AUC of L3=%0.6f'% AUP_all[0])# 颜色，粗细，标签
plt.plot(new_x, y_all[2],'b',lw=1,label='AUC of L3plus=%0.6f'% AUP_all[2])# 颜色，粗细，标签
plt.plot(new_x, y_all[1],'r',lw=1,label='AUC of Sim=%0.6f'% AUP_all[1])# 颜色，粗细，标签
plt.plot(new_x, y_all[3],'y',lw=1,label='AUC of LO=%0.6f'% AUP_all[3])# 颜色，粗细，标签
plt.legend(loc=0) #右下角标注曲线信息
plt.savefig("//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_PR_mycode4.jpg")
plt.show()

##################################################################################################
#保存4种方法的top100，500，1000的平均precision和平均PR曲线的AUC值
results0=['Arabidopsis_thaliana2',w1[99],w2[99],w3[99],w4[99],w1[499],w2[499],w3[499],w4[499],w1[999],w2[999],w3[999],w4[999],AUP_all[0],AUP_all[1],AUP_all[2],AUP_all[3]]
excel_path='//home//chenyu//Sim//info.xls'#文件路径
#excel_path=unicode('D:\\测试.xls','utf-8')#识别中文路径
rbook = xlrd.open_workbook(excel_path,formatting_info=True)#打开文件
wbook = copy(rbook)#复制文件并保留格式
w_sheet = wbook.get_sheet(0)#索引sheet表
row=graphname.index('Arabidopsis_thaliana2')+1
for i in range(len(results0)):
     w_sheet.write(row,i+12,results0[i])
wbook.save(excel_path)#保存文件
print 'results0:',results0
t8=time.time()
print("Overall time:%s"%(t8-t1))

###############################################################################