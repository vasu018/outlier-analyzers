import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import defaultdict
import logging
import random
import scipy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

## PageRank Scores for all the nodes in the network
ranking = [('rt73in01m5cr', 0.03883893607572654),('rt73in00m4cr', 0.03883893607572654),('rt73in13p5ds', 0.027682694166130957),('rt73in12p5ds', 0.027682694166130957),('rt55in00hrcr', 0.025422510183071),('rt75in01hrcr', 0.025422510183071),('st75hr83', 0.019753487693442033),('st75hr82', 0.01971495383713335),('rt75in05hrds', 0.01619138977522491),('rt73ve11m5ar', 0.01613583788402078),('rt73ve10m4ar', 0.01613583788402078),('st55in80hras', 0.01601753069358239),('rt73in10p4ds', 0.01572259786086414),('rt73in11p4ds', 0.01572259786086414),('st73in47p5as', 0.014380165745040824),('st73in46p5as', 0.014380165745040824),('rt55in81hrtn', 0.013895334147276752),('rt55in80hrtn', 0.013895334147276752),('rt55in10hrds', 0.013712161135965925),('rt75in11hrds', 0.013712161135965923),('rt55in04hrds', 0.013330688993924259),('st55in81hras', 0.012293181860751073),('rt55dxce01', 0.011363636363636364),('ts55dxnt01', 0.011363636363636364),('rt73sn15m5ce', 0.011363636363636362),('rt55sn15hrce', 0.011363636363636362),('rt73sn10m4cr', 0.011363636363636362),('rt73sn11m5cr', 0.011363636363636362),('rt75sn14hrce', 0.011363636363636362),('rt55sn11hrcr', 0.011363636363636362),('rt73sn14m4ce', 0.011363636363636362),('rt75sn10hrcr', 0.011363636363636362),('st73in66p5as', 0.01050534096826229),('st73in64p5as', 0.010505340968262288),('st73in67p5as', 0.010505340968262288),('st73in65p5as', 0.010505340968262288),('rt62bs83', 0.010239031416401223),('rt62bs82', 0.010239031416401223),('rt62in83rh', 0.010239031416401222),('rt62in82rh', 0.010239031416401222),('st75hr93', 0.010238550502816888),('st75hr92', 0.010238550502816888),('st55in62hras', 0.01013162935993378),('st55in63hras', 0.01013162935993378),('st55in64hras', 0.01013162935993378),('st55in65hras', 0.01013162935993378),('rt73in70m4as', 0.009920718526828791),('rt73in71m5as', 0.009920718526828791),('rt75in73hrce', 0.008956092830608957),('rt55in72hrce', 0.008956092830608957),('rt73in09m5ce', 0.008814742844707804),('rt73in08m4ce', 0.008814742844707802),('st73in22p5as', 0.008756783076195578),('st73in23p5as', 0.008756783076195578),('st73in20p4as', 0.008756783076195578),('st73in21p4as', 0.008756783076195578),('st73in29p5as', 0.008756783076195578),('st73in28p5as', 0.008756783076195578),('st73in27p4as', 0.008756783076195578),('st73in26p4as', 0.008756783076195578),('st55hr90', 0.008468018572964827),('rt55in70hras', 0.007751864790889417),('rt75in71hras', 0.007751864790889417),('st73in40p4as', 0.007401444808110326),('st73in45p4as', 0.007401444808110326),('st73in44p4as', 0.007401444808110325),('st73in41p4as', 0.007401444808110325),('st73in58p5as', 0.007099269101794488),('st73in42p5as', 0.007099269101794488),('st73in51p5as', 0.007099269101794488),('st73in43p5as', 0.007099269101794487),('st73in59p5as', 0.007099269101794487),('st73in50p5as', 0.007099269101794487),('rt73in04m4ds', 0.006880291225685942),('rt73in05m5ds', 0.006880291225685942),('st73in25p4as', 0.0068802912256859415),('st73in24p4as', 0.0068802912256859415),('rt75hr101', 0.0066977778077574645),('rt75hr102', 0.0066977778077574645),('st75in18hras', 0.006651706747614708),('st55in17hras', 0.006651706747614708),('st75in16hras', 0.006651706747614708),('st55in15hras', 0.006651706747614707),('st55hr73a', 0.006509995325993266),('st73in55p5as', 0.004969292641122629),('st73in54p5as', 0.004969292641122629),('st55hr91', 0.004782203596442194),('st55hr72a', 0.003135974404155674)]
ranks = {}
for node,rank in ranking:
    ranks[node] = rank


#loading the data    
df1 = pd.read_json("signature-ip-access.json", orient='split')
df2 = pd.read_json("signature-route-filter.json", orient='split')
df3 = pd.read_json("signature-route-policy.json", orient='split')

df1 = df1[df1['score_ratio']<=0.7]
df2 = df2[df2['score_ratio']<=0.7]
df3 = df3[df3['score_ratio']<=0.7]

def get_name(outlier):
    return outlier['name']    

df1['outlier_id'] = df1.apply(lambda row: get_name(row['outlier_names']), axis=1)
df2['outlier_id'] = df2.apply(lambda row: get_name(row['outlier_names']), axis=1)
df3['outlier_id'] = df3.apply(lambda row: get_name(row['outlier_names']), axis=1)

df1['feature'] = 'ip_list'
df2['feature'] = 'route_filter'
df3['feature'] = 'route_policy'


df1['Metric_1'] =1- df1['score_ratio']
df2['Metric_1'] =1- df2['score_ratio']
df3['Metric_1'] =1- df3['score_ratio']

df1['Metric_3'] = 0.4075152423812793
df2['Metric_3'] = 0.5116716003557998
df3['Metric_3'] = 0.5822163473976116



#Calculating metric 2
def calc_m2(nodes):
    s = 0
    for node in nodes:
        s = max(s,ranks[node])
    return s             
df['Metric_2'] = df.apply(lambda row: calc_m2(row['outlier_nodes']), axis=1)

x = df[['Metric_2']] #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['Metric_2'] = x_scaled

#Building populations
populations = defaultdict(set)
indices = list(df.index.values)
i=[0]
def build_populations(outlier_id,index):
    populations[outlier_id].add(indices[i[0]])
    i[0]+=1
    
df.apply(lambda row: build_populations(row['outlier_id'],row), axis=1)
x=0
final_pop=defaultdict(set)
for p in populations:
    if len(populations[p])>1:
        final_pop[p] = populations[p]
        
        
#Calculating severity
df['severity_score'] = df['Metric_3'] + df['Metric_2'] + df['Metric_1']


#Assigning severity labels
kmeans = KMeans(n_clusters=4)
df['label1'] = kmeans.fit_predict(df[['severity_score']])
centers = []
for i in kmeans.cluster_centers_:
    centers.append(i[0])

centers_index = np.argsort(centers)[::-1]


label_dict = {}
for i,it in enumerate(centers_index):
    label_dict[it]=i    
centers_index

final_ranks = []
for i in centers:
    final_ranks.append(list(df[df['label']==i].index))

def assign_label(label):
    return 4 - label_dict[label]

df['final_label1'] = df.apply(lambda row: assign_label(row['label']), axis=1)




#functions to update values of final score
def update_score(index,frac):
    df.at[index, 'severity_score'] = df.loc[index]['severity_score']*frac
 
 
#functions to be called when we find a bug or a false positive   
def found_bug(idx,df = df):
    o_id = df.loc[idx]['outlier_id']
    if o_id in final_pop:
        l = final_pop[o_id]
        for n in l:
            update_score(n,1.1)
    o_feature = df.loc[idx]['feature']
    l = feature_pop[o_feature]
    for n in l:
            update_score(n,1.05)
    
    
def found_fp(idx,df=df):
    o_id = df.loc[idx]['outlier_id']
    if o_id in final_pop:
        l = final_pop[o_id]
        for n in l:
            update_score(n,0.9)
    o_feature = df.loc[idx]['feature']
    l = feature_pop[o_feature]
    for n in l:
            update_score(n,0.95)
    
