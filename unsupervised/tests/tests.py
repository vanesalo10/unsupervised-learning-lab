import random
import numpy as np
import sys
sys.path.append("../unsupervised")
from unsupervised.SVD import SVD
from unsupervised.PCA import PCA

random.seed(4)
np.random.seed(4)

n = random.randint(1,10)
m = random.randint(1,10)

A_rectangular = np.random.randint(0,50,(n,m))
print('Rectangular Matrix Ar:\n')
print(A_rectangular)


svd = SVD()
fit = svd.fit(A_rectangular)
comps = fit.transform(A_rectangular)
print('--------------------1c')
print(comps)

svd = SVD(n_components=3)
comps = svd.fit_transform(A_rectangular)
print('--------------------3c')
print(comps)


pca = PCA(n_components=2)
comps = pca.fit_transform(A_rectangular)
print('--------------------3c')
print(comps)