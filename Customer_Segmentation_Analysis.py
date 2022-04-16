# %% [markdown]
# # Customer Segmentation Analysis 
# ### Steps to solve the problem :
# 1. Import Libraries
# 2. Import Dataset
# 3. Data Exploration
# 4. Data Visualization
# 5. K-Means Clustering
# 6. Cluster Selection
# 7. Plot Clusters
# 8. Cluster Analysis

# %% [markdown]
# ## Import Libraries

# %%
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
py.offline.init_notebook_mode(connected = True)

# %%
# import helper module
from Helper_Module_Credit_Risk_Analysis import *
Custom_Helper_Module()

# %% [markdown]
# ## Import Dataset

# %%
# import dataset
df = pd.read_csv('Mall_Customers.csv')

# %% [markdown]
# ## Data Exploration

# %%
df.head()

# %%
df.shape

# %%
df.describe(include='all')

# %%
# CustomerID will not provide us any useful cluster information
# so we can safely drop this column
df.drop('CustomerID', axis = 1, inplace=True)

# %%
df.head()

# %%
Check_Missing_Values(df)

# %%
Check_Feature_Details(df, 'Gender')

# %%
backup_df = df.copy()

# %% [markdown]
# ## Data Visualization

# %%
plt.style.use('fivethirtyeight')

# %% [markdown]
# ### Histograms

# %%
plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    sns.distplot(df[x], bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()

# %% [markdown]
# ### Count Plot of Gender

# %%
plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'Gender' , data = df)
plt.show()

# %%
# keep a backup of the dataframe
backup_df = df.copy()

# %% [markdown]
# ### Distribution of values in Age, Annual Income and Spending Score according to Gender

# %%
plt.figure(1 , figsize = (15 , 7))
n = 0 
for cols in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1 
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.violinplot(x = cols , y = 'Gender' , data = df , palette = 'vlag')
    sns.swarmplot(x = cols , y = 'Gender' , data = df, alpha = 0.5)
    plt.ylabel('Gender' if n == 1 else '')
    plt.title('Violinplot and Swarmplots' if n == 2 else '')
plt.show()

# %% [markdown]
# ### Get Dummies

# %%
# now let's get the dummies for the categorical variables
temp_df = pd.get_dummies(df, drop_first=True)

# %%
dummy_df = temp_df[['Gender_Male', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].copy()

# %%
dummy_df.head()

# %% [markdown]
# ### Ploting the Relation between Gender, Age, Annual Income and Spending Score

# %%
plt.figure(1 , figsize = (16 , 16))
n = 0 
for x in ['Gender_Male', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    for y in ['Gender_Male', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
        n += 1
        plt.subplot(4, 4, n)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        plt.axvline(dummy_df[x].median(), color='g', linestyle='--',  linewidth=2)
        sns.regplot(x = x, y = y, data = dummy_df, scatter_kws = {'alpha': 0.2}, line_kws = {'linewidth':2, 'color': 'red'})
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y)
plt.show()

# %% [markdown]
# ## KMeans Clustering

# %% [markdown]
# ### 1. Segmentation using  Annual Income and Spending Score

# %%
df = backup_df.copy()

# %%
'''Annual Income and spending Score'''
column_list_for_kmeans = ['Annual Income (k$)', 'Spending Score (1-100)']

# %%
X = df[column_list_for_kmeans].copy()
X.head()

# %%
standard_scaler = StandardScaler()
X[column_list_for_kmeans] = standard_scaler.fit_transform(X[column_list_for_kmeans])


# %%
inertia = []
for n in range(1 , 11):
    kmeans = KMeans(n_clusters = n, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# %%
kl = KneeLocator(list(range(1 , 11)), inertia, S=1.0, curve='convex', direction='decreasing')
optimum_cluster = kl.knee
print('Optimum number of clusters: ', optimum_cluster)

# %%
plt.figure(1, figsize = (15 ,6))
plt.plot(np.arange(1, 11) , inertia , 'o')
plt.plot(np.arange(1, 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.vlines(optimum_cluster, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', linewidth=2, color='black')
plt.show()

# %%
kmeans = KMeans(n_clusters = optimum_cluster, init = 'k-means++', random_state = 42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# now, lets calculate the silhouette score of this model
model_1_silhouette_score = silhouette_score(X, kmeans.labels_, metric='euclidean')
print('Silhouette score: {:.2f}'.format(model_1_silhouette_score))

# %%
# add the cluster label and inverse scalar transformation
df['Cluster'] = labels.tolist()

# %%
# now lets make the centroid dataframe
centroids_df_column_names = ['centroid_x', 'centroid_y']
centroids_df = pd.DataFrame(data = centroids, columns = centroids_df_column_names)
centroids_df[centroids_df_column_names] = standard_scaler.inverse_transform(centroids_df[centroids_df_column_names])

# %%
df.head()

# %%
centroids_df.head()

# %%
x_feature = 'Annual Income (k$)'
y_feature = 'Spending Score (1-100)'
x_centroid = 'centroid_x'
y_centroid = 'centroid_y'

# %%
h = 0.01
x_min, x_max = df[x_feature].min() - 1, df[x_feature].max() + 1
y_min, y_max = df[y_feature].min() - 1, df[y_feature].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]) 

# %%
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z, interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin = 'lower')

plt.scatter( x = x_feature, y = y_feature, data = df, c = labels, s = 100)
plt.scatter(x = centroids_df[x_centroid].values, y = centroids_df[y_centroid].values, s = 200, c = 'red', alpha = 0.5)
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.show()

# %% [markdown]
# ### 2. Segmentation using Age, Annual Income and Spending Score

# %%
df = backup_df.copy()

# %%
'''Age, Annual Income and spending Score'''
column_list_for_kmeans = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# %%
X = df[column_list_for_kmeans].copy()
X.head()

# %%
standard_scaler = StandardScaler()
X[column_list_for_kmeans] = standard_scaler.fit_transform(X[column_list_for_kmeans])


# %%
inertia = []
for n in range(1 , 11):
    kmeans = KMeans(n_clusters = n, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# %%
kl = KneeLocator(list(range(1 , 11)), inertia, S=1.0, curve='convex', direction='decreasing')
optimum_cluster = kl.knee
print('Optimum number of clusters: ', optimum_cluster)

# %%
plt.figure(1, figsize = (15 ,6))
plt.plot(np.arange(1, 11) , inertia , 'o')
plt.plot(np.arange(1, 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.vlines(optimum_cluster, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', linewidth=2, color='black')
plt.show()

# %%
kmeans = KMeans(n_clusters = optimum_cluster, init = 'k-means++', random_state = 42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# now, lets calculate the silhouette score of this model
model_2_silhouette_score = silhouette_score(X, kmeans.labels_, metric='euclidean')
print('Silhouette score: {:.2f}'.format(model_2_silhouette_score))

# %%
# add the cluster label and inverse scalar transformation
df['Cluster'] = labels.tolist()


# %%
# now lets make the centroid dataframe
centroids_df_column_names = ['centroid_x', 'centroid_y', 'centroid_z']
centroids_df = pd.DataFrame(data = centroids, columns = centroids_df_column_names)
centroids_df[centroids_df_column_names] = standard_scaler.inverse_transform(centroids_df[centroids_df_column_names])

# %%
df.head()

# %%
centroids_df.head()

# %%
# 3d scatterplot using plotly
Scene = dict(xaxis = dict(title = 'Age'), yaxis = dict(title = 'Annual Income'),zaxis = dict(title = 'Spending Score'))
trace_0 = go.Scatter3d(x=df['Age'], y=df['Annual Income (k$)'], z=df['Spending Score (1-100)'], mode='markers', marker=dict(color = labels, size = 5, line=dict(color= labels, width = 10)), name='Data')
trace_1 = go.Scatter3d(x=centroids_df['centroid_x'], y=centroids_df['centroid_y'], z=centroids_df['centroid_z'], mode='markers', marker=dict(color = 'Black', size = 10), name='Centroid')
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace_0, trace_1]
fig = go.Figure(data = data, layout = layout)
fig.show()

# %% [markdown]
# ### 3. Segmentation using Gender, Age, Annual Income and Spending Score

# %%
df = backup_df.copy()

# %%
'''Gender, Age, Annual Income and spending Score'''
column_list_for_kmeans = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
column_list_for_scaling = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# %%
X = df[column_list_for_kmeans].copy()
X.head()

# %%

standard_scaler = StandardScaler()
X[column_list_for_scaling] = standard_scaler.fit_transform(X[column_list_for_scaling])


# %%
# now let's get the dummies for the categorical variables
X = pd.get_dummies(X, drop_first=True)

# %%
inertia = []
for n in range(1 , 11):
    kmeans = KMeans(n_clusters = n, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# %%
kl = KneeLocator(list(range(1 , 11)), inertia, S=1.0, curve='convex', direction='decreasing')
optimum_cluster = kl.knee
print('Optimum number of clusters: ', optimum_cluster)

# %%
plt.figure(1, figsize = (15 ,6))
plt.plot(np.arange(1, 11) , inertia , 'o')
plt.plot(np.arange(1, 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.vlines(optimum_cluster, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', linewidth=2, color='black')
plt.show()

# %%
kmeans = KMeans(n_clusters = optimum_cluster, init = 'k-means++', random_state = 42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# now, lets calculate the silhouette score of this model
model_3_silhouette_score = silhouette_score(X, kmeans.labels_, metric='euclidean')
print('Silhouette score: {:.2f}'.format(model_3_silhouette_score))

# %%
# add the cluster label and inverse scalar transformation
df['Cluster'] = labels.tolist()


# %%
df.head()

# %%
# 3d scatterplot using plotly
Scene = dict(xaxis = dict(title = 'Age'), yaxis = dict(title = 'Annual Income'),zaxis = dict(title = 'Spending Score'))
trace = go.Scatter3d(x=df['Age'], y=df['Annual Income (k$)'], z=df['Spending Score (1-100)'], mode='markers', marker=dict(color = labels, size = 5, line=dict(color= labels, width = 10)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()

# %%
model_3_df = df.copy()
print('Silhouette score: {:.2f}'.format(model_3_silhouette_score))

# %% [markdown]
# ### 4. Segmentation using PCA

# %%
df = backup_df.copy()

# %%
'''Gender, Age, Annual Income and spending Score'''
column_list_for_kmeans = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
column_list_for_scaling = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# %%
X = df[column_list_for_kmeans].copy()
X.head()

# %%
standard_scaler = StandardScaler()
X[column_list_for_scaling] = standard_scaler.fit_transform(X[column_list_for_scaling])

# %%
# now let's get the dummies for the categorical variables
X = pd.get_dummies(X, drop_first=True)

# %%
X.head()

# %%
# pca = PCA(n_components=5)
pca = PCA()
principalComponents = pca.fit_transform(X)

features = range(pca.n_components_)
variance_ratio = pca.explained_variance_ratio_
print(variance_ratio)
plt.bar(features, variance_ratio)
plt.xlabel('PCA Feature')
plt.ylabel('Variance %')
plt.xticks(features)
# Show the plot
plt.show()

PCA_components = pd.DataFrame(principalComponents)

# %%
# first 3 PCA features explain > 80% of the data variance
# so we will build model with first 3 princiapl components
selected_PCA_components = PCA_components.iloc[:,:2]
print(selected_PCA_components)

# %%
inertia = []
for n in range(1 , 11):
    kmeans = KMeans(n_clusters = n, init = 'k-means++', random_state = 42)
    kmeans.fit(selected_PCA_components)
    inertia.append(kmeans.inertia_)

# %%
kl = KneeLocator(list(range(1 , 11)), inertia, S=1.0, curve='convex', direction='decreasing')
optimum_cluster = kl.knee
print('Optimum number of clusters: ', optimum_cluster)

# %%
plt.figure(1, figsize = (15 ,6))
plt.plot(np.arange(1, 11) , inertia , 'o')
plt.plot(np.arange(1, 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.vlines(optimum_cluster, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', linewidth=2, color='black')
plt.show()

# %%
kmeans = KMeans(n_clusters = optimum_cluster, init = 'k-means++', random_state = 42)
kmeans.fit(selected_PCA_components)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# now, lets calculate the silhouette score of this model
model_4_silhouette_score = silhouette_score(selected_PCA_components, kmeans.labels_, metric='euclidean')
print('Silhouette score: {:.2f}'.format(model_4_silhouette_score))

# %%
# add the cluster label and inverse scalar transformation
df['Cluster'] = labels.tolist()


# %%
df.head()

# %%
# 3d scatterplot using plotly
Scene = dict(xaxis = dict(title = 'Age'), yaxis = dict(title = 'Annual Income'),zaxis = dict(title = 'Spending Score'))
trace = go.Scatter3d(x=df['Age'], y=df['Annual Income (k$)'], z=df['Spending Score (1-100)'], mode='markers', marker=dict(color = labels, size = 5, line=dict(color= labels, width = 10)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene,height = 800,width = 800)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()

# %%
model_4_df = df.copy()
print('Silhouette score: {:.2f}'.format(model_4_silhouette_score))

# %% [markdown]
# ## Cluster Analysis

# %%
model_4_df.head()

# %%
df = model_4_df.copy()

# %%
# to compare attributes of the different clusters, lets find the 
# average of all variables across each cluster
median_df = df.groupby(['Cluster'], as_index=False).median()
median_df

# %%
gender_df = pd.DataFrame(df.groupby(['Cluster','Gender'])['Gender'].count())
gender_df

# %%
male_df = df.groupby('Cluster')['Gender'].apply(lambda x: x[x == 'Male'].count())
female_df = df.groupby('Cluster')['Gender'].apply(lambda x: x[x == 'Female'].count())
female_percentage = round(female_df*100/(male_df+female_df), 2)
gender_perecentage_df = pd.DataFrame({'Cluster':female_percentage.index, 'Female(%)':female_percentage.values})
gender_perecentage_df


# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Cluster Analysis')

# Annual Income
sns.barplot(ax=axes[0, 0], x='Cluster', y='Annual Income (k$)', data=median_df)

# Spending Score
sns.barplot(ax=axes[0, 1], x='Cluster',y='Spending Score (1-100)', data=median_df)

# Gender
sns.barplot(ax=axes[1, 0], x='Cluster',y='Female(%)', data=gender_perecentage_df)

# Age
sns.barplot(ax=axes[1, 1], x='Cluster', y='Age', data=median_df)

# Show the plot
plt.show()

# %%



