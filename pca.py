import numpy as np


def pca(x, num_components):

    #Step-1
    x_mean = np.mean(x, axis=0)  #np.zeros(num_components)  #
    x_meaned = x - x_mean

    #Step-2
    cov_mat = np.cov(x_meaned, rowvar=False)

    #Step-3
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    #Step-5
    eigenvector_subset = sorted_eigenvectors[:, :num_components]

    #Step-6
    transform = lambda x: np.dot(eigenvector_subset.transpose(), x.transpose()).transpose()
    x_reduced = transform(x_meaned)
    inverse_transform = lambda pc: np.dot(eigenvector_subset, pc.transpose()).transpose() + x_mean
    print(x[:1, :], inverse_transform(x_reduced[:1, :]))

    return x_reduced, transform, inverse_transform

# import pandas as pd
#
# #Get the IRIS dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# data = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
#
# #prepare the data
# x = data.iloc[:,0:4]
#
# #prepare the target
# target = data.iloc[:,4]
#
# #Applying it to PCA function
# mat_reduced = pca(x, 2)
#
# #Creating a Pandas DataFrame of reduced Dataset
# principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])
#
# #Concat it with target variable to create a complete Dataset
# principal_df = pd.concat([principal_df , pd.DataFrame(target)] , axis = 1)
#
# import matplotlib.pyplot as plt
#
# plt.figure(figsize = (6,6))
# principal_df['color'] = '#1f77b4'
# principal_df.loc[principal_df['target'] == 'Iris-versicolor', 'color'] = '#ff7f0e'
# principal_df.loc[principal_df['target'] == 'Iris-virginica', 'color'] = '#2ca02c'
# plt.scatter(principal_df['PC1'], principal_df['PC2'], c=principal_df['color'])  #, s = 60 , palette= 'icefire')
# plt.show()

if __name__ == '__main__':
    from read_data.dowa import read_data
    wind_data = read_data({'name': 'mmij'})
    from preprocess_data import preprocess_data
    wind_data = preprocess_data(wind_data)
    pca(wind_data['training_data'], 2)