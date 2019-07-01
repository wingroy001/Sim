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
#�����￴��ʼ�滻�ű��е���������


print 'Network: Arabidopsis_thaliana2'
G0=nx.read_weighted_edgelist('//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2.txt') #��ȡԭͼ
print 'nx.info(G0)'
print nx.info(G0)
print 'average clustering coefficient:', nx.average_clustering(G0)
nx.write_weighted_edgelist(G0, '//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_00.txt')
G0=nx.read_weighted_edgelist('//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_00.txt') #���¶�ȡԭͼ    
G1=nx.empty_graph(G0.nodes()) #��ʼ����һ�ֵ������Լ�Ϊ��ͼ
samplesize=int(0.1*len(G0.edges))  #�������Լ��ܹ�ģ=samplesize*2
iteration=10
print 'iteration=',iteration
for i in range(iteration):  
  edges1 = random.sample(G0.edges(), samplesize) #ԭͼ�����ȡ10%����    
  G0.remove_edges_from(edges1) #ɾ����һ��ѡ���10%�ı�
  G0.add_edges_from(G1.edges()) #�����һ��ɾ����10%�ıߣ���֤G0ӵ��90%�ı�
  nx.write_weighted_edgelist(G0, '//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_01_%d.txt'%(i+1)) #ѵ����
  G1=nx.empty_graph(G0.nodes())  #��ʼ����һ�ֵ������Լ�Ϊ��ͼ
  G1.add_edges_from(edges1)  
  nx.write_weighted_edgelist(G1, '//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_02_%d.txt'%(i+1))#������ 
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
                             create_using=None, nodetype=None, encoding='utf-8')#ѵ����
  G1= nx.read_weighted_edgelist('//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_02_%d.txt'%(kfold+1), delimiter=' ', 
                             create_using=None, nodetype=None, encoding='utf-8')#������ 
  G2= nx.complement(G)
  edges=[i for i in G1.edges] #ȡ����������
  G2.remove_edges_from(edges)  #��ѵ��ͼ�Ĳ�ͼ��ɾȥ����������Ϊ������
  #G0ԭʼͼ��Gѵ������G1��������G2������
  n=len(G0.nodes()) #ԭʼͼ�ڵ���
  m=len(G0.edges)   #ԭʼͼ����
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
    labels1.append(1)  #��������ǩ��1
    scores3.append(aa) #L3+��������ֵ
    scores1.append(bb) #L3��������ֵ
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
    labels1.append(2)  #��������ǩ��2
    scores3.append(aa) #L3+��������ֵ   
    scores1.append(bb) #L3��������ֵ
###############################################################################    
  print 'L3 is done'
  t3=time.time()
  print("time:%s"%(t3-t2))
  print 'Sim processing...'
###############################################################################
  #Sim
  def jacc2(u, v):  # �������ƶ�Ϊ����ͬ�ھ��������ھ����еı��������ƶ�Ϊ1��ʾ��������һģһ�� 
    w=[u[i]+v[i] for i in range(n)] #u,v������ά��n��Ҫ�ȶ��壬wΪu��v��Ӧ����֮��
    j=n-w.count(0) #u��v���ܹ��ھ���
    if j!=0:
       jac=w.count(2)/j 
    else:  #������������0����
       jac=1  
    return jac 
###############################################################################
  index=[i for i in G.nodes()] #����G�Ľڵ㼯Ϊ�б��ʽ����ԭ�ڵ㼯�нڵ�˳��һ��   

  X=np.zeros((n,n)) #��ʼ���ڽӾ���

  for i in range(n):
    for j in range(i+1,n):
        if (index[i],index[j]) in G.edges:
            X[i,j],X[j,i]=1,1
  #print 'X='
  #print X       #�ڽӾ���   
  #print jacc2(X[0], X[1])
  Y=np.zeros((n,n))  #�ܿ������ƶȾ����ʼ��Ϊ�����
  Z=np.zeros((n,n))  #���Ӹ��ʾ����ʼ��Ϊ�����
###############################################################################
  #jaccard=[]
  for i in range(n):
    for j in range(i,n):
        J=jacc2(X[i], X[j]) 
        Y[j,i],Y[i,j]=J,J
  Z=np.dot(X,Y)+np.dot(Y,X)   
  #print Y  #�ܿ������ƶȾ���
  #print Z  #���Ӹ��ʾ���   
  #jaccard.sort(reverse=True)   #�÷ְ��Ӵ�С������� 
  #np.savetxt("//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_Sim_jaccard.txt",jaccard,fmt='%s')
  order,k=dict(),0     #order�ֵ��ʽ���洢�ڵ���ȣ��ڽڵ㼯�е�˳��     
  for i in G.nodes():
    order[i]=k
    k+=1  
   
############################################################################### 
  scores2,simscore= [],[]
  for (i, j, w) in G2.edges(data=True):
    scores2.append(Z[order[i],order[j]]) #�������÷�
    #simscore.append((Z[order[i],order[j]],i,j,1))

  for (i, j, w) in G1.edges(data=True):
    scores2.append(Z[order[i],order[j]]) #�������÷�
    #simscore.append((Z[order[i],order[j]],i,j,2) )
    
  #simscore.sort(reverse=True)   #�÷ְ��Ӵ�С�������
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
  I = np.eye(n) #��λ����
  alpha=0.00001  
  f=X-np.dot(X,np.linalg.inv(I+alpha*np.dot(X,X)))  #X���ڽӾ���f�����Ӹ��ʾ���
  scores4=[]
  for (i, j, w) in G2.edges(data=True):
    scores4.append(f[order[i],order[j]]) #�������÷�
  for (i, j, w) in G1.edges(data=True):
    scores4.append(f[order[i],order[j]]) #�������÷�
###############################################################################
  t5=time.time()
  print 'LO is done'
  print("time:%s"%(t5-t4))
  print 'Top k precision processing...'
###############################################################################
  #top k  
  x=zip(scores1,labels1) #��score��label���  L3
  x.sort(reverse=True) #��score�Ӵ�С����label��������
  y=zip(scores2,labels1) #��score��label��� Sim
  y.sort(reverse=True) #��score�Ӵ�С����label��������
  x1=zip(scores3,labels1) #��score��label���  L3Plus
  x1.sort(reverse=True) #��score�Ӵ�С����label��������
  z=zip(scores4,labels1) #��score��label���  LO
  z.sort(reverse=True) #��score�Ӵ�С����label��������
  w1,w2,w3,w4=[],[],[],[] #��ʼ��precision
  pos1,pos2,pos3,pos4,k=0,0,0,0,min(1000,len(labels1)) #ǰk��Ԥ�����������ĸ���
  for i in range(k):
    pos1+=int(x[i][1]/2)  #��������ǩ��2����������ǩ��1����ǩ����2ȡ��֮�ͼ�Ϊ����������
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
    plt.figure(k) #ͼ�ı��
    plt.figure(figsize=(8,6)) #������С
    plt.title("Arabidopsis_thaliana2" ) #ͼ������ǩ
    plt.xlabel("Top %d prediction"%k) #�������ǩ
    plt.ylabel("Precision")  #�������ǩ
    plt.ylim(0,1) #�����귶Χ
    plt.xlim(0,k) #�����귶Χ
    plt.plot(x, y1,'g',lw=1,label='AUC of L3=%0.6f'% AUP1)# ��ɫ����ϸ����ǩ
    plt.plot(x, y3,'b',lw=1,label='AUC of L3plus=%0.6f'% AUP3)# ��ɫ����ϸ����ǩ
    plt.plot(x, y2,'r',lw=1,label='AUC of Sim=%0.6f'% AUP2)# ��ɫ����ϸ����ǩ
    plt.plot(x, y4,'y',lw=1,label='AUC of LO=%0.6f'% AUP4)# ��ɫ����ϸ����ǩ
    plt.legend(loc=0) #���½Ǳ�ע������Ϣ
    plt.savefig("//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_%d_Top%d_Precision.jpg"%(kfold+1,k)) #����ROC����ͼ
    plt.show()# show the plot on the screen  
  yy1.append(y1) #���淽��L3ǰ1000��Ԥ���precision
  yy2.append(y2) #���淽��Simǰ1000��Ԥ���precision
  yy3.append(y3) #���淽��L3+ǰ1000��Ԥ���precision
  yy4.append(y4) #���淽��LOǰ1000��Ԥ���precision
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
  plt.figure(figsize=(8,6)) #������С
  plt.title("Arabidopsis_thaliana2" ) #ͼ������ǩ
  plt.xlabel('Recall')# make axis labels
  plt.ylabel('Precision')
  plt.ylim(0,1) #�����귶Χ
  plt.xlim(0,1) #�����귶Χ
  plt.plot(recall1, precision1,'g',lw=1,label='AUC of L3=%0.6f'% AUP1)
  plt.plot(recall3, precision3,'b',lw=1,label='AUC of L3plus=%0.6f'% AUP3)
  plt.plot(recall2, precision2,'r',lw=1,label='AUC of Sim=%0.6f'% AUP2)
  plt.plot(recall4, precision4,'y',lw=1,label='AUC of LO=%0.6f'% AUP4)
  plt.legend(loc=0) #���½Ǳ�ע������Ϣ
  plt.savefig("//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_%d_PR_mycode3.jpg"%(kfold+1))
  plt.show()
  print 'Precision recall cure is done'
  t7=time.time()
  print("time:%s"%(t7-t6))

###############################################################################
w1,w2,w3,w4=[0]*1000,[0]*1000,[0]*1000,[0]*1000 #��ʼ��ǰ1000��Ԥ���ƽ��precision���ܹ�Ԥ������ı�����ڻ����1000�Σ����򱨴�
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
    plt.figure(k+1) #ͼ�ı��
    plt.figure(figsize=(8,6)) #������С
    plt.title("Arabidopsis_thaliana2" ) #ͼ������ǩ
    plt.xlabel("Top %d prediction"%k) #�������ǩ
    plt.ylabel("Precision")  #�������ǩ
    plt.ylim(0,1) #�����귶Χ
    plt.xlim(0,k) #�����귶Χ
    plt.plot(x, y1,'g',lw=1,label='AUC of L3=%0.6f'% AUP1)# ��ɫ����ϸ����ǩ
    plt.plot(x, y3,'b',lw=1,label='AUC of L3plus=%0.6f'% AUP3)# ��ɫ����ϸ����ǩ
    plt.plot(x, y2,'r',lw=1,label='AUC of Sim=%0.6f'% AUP2)# ��ɫ����ϸ����ǩ
    plt.plot(x, y4,'y',lw=1,label='AUC of LO=%0.6f'% AUP4)# ��ɫ����ϸ����ǩ
    plt.legend(loc=0) #���½Ǳ�ע������Ϣ
    plt.savefig("//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_Top%d_Precision.jpg"%(k)) # ����ƽ��ROC����ͼ
    plt.show()# show the plot on the screen 
 
##################################################################################################
new_length=500 #�ز����ĵ���
new_x = np.linspace(0, 1, new_length) #�µĺ�����new_length���㣬��Χ��0��1�����Ȳɵ�
new_y=[0]*new_length #��ʼ��ƽ�����ߵ�������
y_all,AUP_all=[],[] #��ʼ��4�ַ�����ƽ��PR�����������ƽ��AUCֵ
for j in range(4):  #4�ַ�����˳��ΪL3,Sim��L3+��LO
  for i in range(iteration):
    tem_y = interpolate.interp1d(Rec[i][j], Pre[i][j])  
    new_y=[new_y[i]+tem_y(new_x)[i] for i in range(new_length)] #�������߶�Ӧ���������
  new_y=[new_y[i]/iteration for i in range(new_length)] #�������߶�Ӧ�������ƽ��ֵΪƽ�����ߵ�������
  AUP_all.append(auc(new_x, new_y))
  y_all.append(new_y)
plt.figure(11)
plt.figure(figsize=(8,6)) #������С
plt.title("Arabidopsis_thaliana2" ) #ͼ������ǩ
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')
plt.ylim(0,1) #�����귶Χ
plt.xlim(0,1) #�����귶Χ
plt.plot(new_x, y_all[0],'g',lw=1,label='AUC of L3=%0.6f'% AUP_all[0])# ��ɫ����ϸ����ǩ
plt.plot(new_x, y_all[2],'b',lw=1,label='AUC of L3plus=%0.6f'% AUP_all[2])# ��ɫ����ϸ����ǩ
plt.plot(new_x, y_all[1],'r',lw=1,label='AUC of Sim=%0.6f'% AUP_all[1])# ��ɫ����ϸ����ǩ
plt.plot(new_x, y_all[3],'y',lw=1,label='AUC of LO=%0.6f'% AUP_all[3])# ��ɫ����ϸ����ǩ
plt.legend(loc=0) #���½Ǳ�ע������Ϣ
plt.savefig("//home//chenyu//Sim//Arabidopsis_thaliana2//Arabidopsis_thaliana2_PR_mycode4.jpg")
plt.show()

##################################################################################################
#����4�ַ�����top100��500��1000��ƽ��precision��ƽ��PR���ߵ�AUCֵ
results0=['Arabidopsis_thaliana2',w1[99],w2[99],w3[99],w4[99],w1[499],w2[499],w3[499],w4[499],w1[999],w2[999],w3[999],w4[999],AUP_all[0],AUP_all[1],AUP_all[2],AUP_all[3]]
excel_path='//home//chenyu//Sim//info.xls'#�ļ�·��
#excel_path=unicode('D:\\����.xls','utf-8')#ʶ������·��
rbook = xlrd.open_workbook(excel_path,formatting_info=True)#���ļ�
wbook = copy(rbook)#�����ļ���������ʽ
w_sheet = wbook.get_sheet(0)#����sheet��
row=graphname.index('Arabidopsis_thaliana2')+1
for i in range(len(results0)):
     w_sheet.write(row,i+12,results0[i])
wbook.save(excel_path)#�����ļ�
print 'results0:',results0
t8=time.time()
print("Overall time:%s"%(t8-t1))

###############################################################################