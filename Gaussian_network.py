import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

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
print("first list is")
print(multiClassEncodedList)
l1 = list(multiClassEncodedList)

clf = mixture.GaussianMixture(n_components=1, covariance_type='full')
#clf.fit(X_train)
clf.fit(multiClassEncodedList)
Z = -clf.score_samples(np.array(multiClassEncodedList))
print("the scores given by GMM when components is 1 are as follows:")
print(Z)

clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
#clf.fit(X_train)
clf.fit(multiClassEncodedList)
Z = -clf.score_samples(np.array(multiClassEncodedList))
print("the scores given by GMM when components is 2 are as follows:")
print(Z)


clf = mixture.GaussianMixture(n_components=3, covariance_type='full')
#clf.fit(X_train)
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
print("second list is")
print(multiClassEncodedList2)
l2 =list(multiClassEncodedList2)


list3 = [list(a) for a in zip(l1, l2)]
for i in range(len(list3)):
    print("values are")
    print(list3[i][0])
print("concatented list is")
print(list3)

for i in range(len(list3)):
    list3[i][0]=list3[i][0].tolist()
    list3[i][1] = list3[i][1].tolist()

print(list3)

for i in range(len(list3)):
    list3[i]=list3[i][0] + list3[i][1]


print(list3)
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(list3)
#clf.fit(list3)
Z = -clf.score_samples(np.array(list3))
print("log likeloihood scores are")
print(Z)


