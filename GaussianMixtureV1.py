import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 500

# generate random sample, having two components
np.random.seed(0)

#Generate the first Gaussian sample
# generate spherical data centered on (30, 30)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([30, 30])
print("shifted gaussian is")
print(shifted_gaussian)

#Generate the second Gaussian sample
# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
print("stretched gaussian is")
print(stretched_gaussian)

# In order to get the mean and variance for the main Gaussian mixture, merge the 2 samples
# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])
print("Training set is")
print(X_train)

# fit a Gaussian Mixture Model with two components
#Now, this mixture of 2 samples is like the training data for the GMM
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

#Test this model on new data by plotting negative log-likelihood
# display predicted scores by the model as a contour plot
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
