# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


## Prepare Data
# load Iris dataset into dataframe
df = pd.read_csv(filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
                 , header=None, sep=',')
df.columns=['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

# split data table into data X and class labels y
X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

# fit data onto an unit scale (mean=0, variance=1)
X_std = StandardScaler().fit_transform(X)
df = pd.DataFrame(data = X_std, columns = ['sepal length', 'sepal width', 'petal length', 'petal width'])


## Calculate Principal Components
# calculate the covariance matrix from the standardised data
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs.round(2))
print('\nEigenvalues \n%s' %eig_vals.round(2))

# perform SVD to calculate the eigenvectors
u,s,v = np.linalg.svd(X_std.T)
u.round(2)


## Sort the Eigenpairs
# make a list of eigenpair tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# sort the eigenpair tuples in descending order by eigenvalues
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# confirm the eigenvalues are sorted correctly
print('\nEigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


## Visualisation of Explained Variance
# calculate the cumulative variance
total = sum(eig_vals)
var = [(i / total)*100 for i in eig_vals]
cum_var = np.cumsum(var)

# bar chart of explained variance
with plt.style.context('seaborn-pastel'):
    g = plt.figure(figsize=(8, 6))
    
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    plt.bar(range(4), var, alpha=1, align='center',
            label='individual variance')
    plt.step(range(4), cum_var, where='mid',
             label='cumulative variance')
    
    plt.title('Explained Variance', fontsize=12)
    plt.ylabel('Ratio of explained variance', fontsize=12)
    plt.xlabel('Principal components', fontsize=12)
    plt.legend(loc='best')
    
    plt.show()

    g.savefig("exp_var.pdf", bbox_inches='tight')


## Projection to 2D
# construct the projection matrix W from the top two eigenvectors
a = np.array([-u[0][0], u[0][1]])
b = np.array([-u[1][0], u[1][1]])
c = np.array([-u[2][0], u[2][1]])
d = np.array([-u[3][0], u[3][1]])

w_matrix = np.stack((a,b,c,d))

print("\nMatrix W:\n", w_matrix)

# calculate dot product of matrix X (standardized) and W to get the new subspace feature matrix Y
Y = X_std.dot(w_matrix)

# scatter plot of dataset in terms of two principal components
with plt.style.context('seaborn-pastel'):
    
    f = plt.figure(figsize=(6, 6))
    plt.rc('font', family='serif')
    
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['cyan', 'pink', 'violet']
    for lab, col in zip(classes, colors):
        plt.scatter(Y[y==lab, 0],
                    Y[y==lab, 1],
                    label=lab,
                    c=col)
        
    plt.title('Projection of Iris dataset to 2D', fontsize=11)
    plt.xlabel('First Principal Component', fontsize=11)
    plt.ylabel('Second Principal Component', fontsize=11)
    plt.legend(loc='best')
    
    plt.show()
    
    f.savefig("Iris_2D.pdf", bbox_inches='tight')


## Reconstruction of data using two principal components
# reconstruct the original dataset using two principal components
mu = np.mean(X, axis=0)
pca = PCA(n_components=2)
Y_pca = pca.fit(X)

nComp = 2
Xhat = np.dot(pca.transform(X)[:,:nComp], pca.components_[:nComp,:])
Xhat += mu

# scatter plot of reconstructed data using two principal components
with plt.style.context('seaborn-pastel'):
    h = plt.figure(figsize=(6, 6))
    plt.rc('font', family='serif')
    
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['cyan', 'pink', 'violet']
    for lab, col in zip(classes, colors):
        plt.scatter(Xhat[y==lab, 0],
                    Xhat[y==lab, 1],
                    label=lab,
                    c=col)
    
    plt.title('Reconstruction of Iris data using two principal components', fontsize=12)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(loc='best')
    
    plt.show()
    
    h.savefig("Iris_recon.pdf", bbox_inches='tight')