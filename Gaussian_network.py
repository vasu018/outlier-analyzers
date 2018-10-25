import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import math

multiclass_feature_Xi = [("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "3.3.3.3"),
                      ("3.3.3.3", ""),
                      ("1.1.1.1", ""),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "3.3.3.3"),
                      ("1.1.1.1", "3.3.3.3"),
                      ("1.1.1.1", ""),
                      ("1.1.1.1", "3.3.3.3"),
                      (),
                      ("", ""),
                      ("1.1.1.1", ""),
                      ("2.2.2.2", "")]


one_hot_multiclass = MultiLabelBinarizer()
multiClassEncodedList = one_hot_multiclass.fit_transform(multiclass_feature_Xi)
print("first feature encoded is")
print(multiClassEncodedList)
feature_1 = list(multiClassEncodedList)

#Gaussian Mixture is used for soft clustering. Insted of assigning points to specific classes it assigns probability.
#The n_components parameter in the Gaussian is used to specify the number of Gaussians.
clf = mixture.GaussianMixture(n_components=1, covariance_type='full')
clf.fit(multiClassEncodedList)
Z = -clf.score_samples(np.array(multiClassEncodedList))
print("the scores given by GMM when components is 1 are as follows:")
print(Z)

clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(multiClassEncodedList)
Z = -clf.score_samples(np.array(multiClassEncodedList))
print("the scores given by GMM when components is 2 are as follows:")
print(Z)


clf = mixture.GaussianMixture(n_components=3, covariance_type='full')
clf.fit(multiClassEncodedList)
Z = -clf.score_samples(np.array(multiClassEncodedList))
print("the scores given by GMM when components is 3 are as follows:")
print(Z)


NTP_Servers = [("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "3.3.3.3"),
                      ("3.3.3.3", ""),
                      ("1.1.1.1", "4.4.4.4"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "2.2.2.2"),
                      ("1.1.1.1", "5.5.5.5"),
                      ("1.1.1.1", "3.3.3.3"),
                      ("1.1.1.1", ""),
                      ("1.1.1.1", "3.3.3.3"),
                      (),
                      ("", ""),
                      ("1.1.1.1", ""),
                      ("2.2.2.2", "")]

one_hot_multiclass = MultiLabelBinarizer()
multiClassEncodedList2 = one_hot_multiclass.fit_transform(NTP_Servers)
print("second feature encoded is")
print(multiClassEncodedList2)
feature_2 =list(multiClassEncodedList2)


concatenated_features= [list(a) for a in zip(feature_1, feature_2)]
for i in range(len(concatenated_features)):
    print("values are")
    print(concatenated_features[i][0])
print("concatented features are")
print(concatenated_features)

for i in range(len(concatenated_features)):
    concatenated_features[i][0]=concatenated_features[i][0].tolist()
    concatenated_features[i][1] = concatenated_features[i][1].tolist()



for i in range(len(concatenated_features)):
    concatenated_features[i]=concatenated_features[i][0] + concatenated_features[i][1]



clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(concatenated_features)
Z = -clf.score_samples(np.array(concatenated_features))
print("log likeloihood scores are")
print(Z)


