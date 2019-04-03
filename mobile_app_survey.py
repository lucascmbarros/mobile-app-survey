# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:20:09 2019

@author: lucas
"""
# Importing new libraries
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis
from sklearn.cluster import KMeans # k-means clustering


# Importing known libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


##############################################################################
#                           CODE FOR DATA ANALYSIS
##############################################################################

survey_df = pd.read_excel('mobile_app_survey.xlsx')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

survey_df.columns

for col in enumerate(survey_df):
    print(col)


#######################################################
# Step: 1 Data preparation
#######################################################

# Reversing Questions

survey_df['q12'].value_counts()


# Create a new column

survey_df['rev_q12'] = -100

survey_df['rev_q12'][survey_df['q12'] == 1] = 6
survey_df['rev_q12'][survey_df['q12'] == 2] = 5
survey_df['rev_q12'][survey_df['q12'] == 3] = 4
survey_df['rev_q12'][survey_df['q12'] == 4] = 3
survey_df['rev_q12'][survey_df['q12'] == 5] = 2
survey_df['rev_q12'][survey_df['q12'] == 6] = 1

survey_df['rev_q12'].value_counts()

# Creating a new column with the number of children

survey_df['n_children'] = (survey_df.q50r1 + survey_df.q50r2 +
                           survey_df.q50r3 + survey_df.q50r4 +
                           survey_df.q50r5)
# Drop old column

survey_df = survey_df.drop(columns = ['q12'],
               axis = 1)


# Droping columns that indicate demography

survey_reduced = survey_df.drop(columns = ['caseID', 'q1',
                                           'q48', 'q49', 'q50r1',
                                           'q50r2', 'q50r3', 'q50r4',
                                           'q50r5', 'q54', 'q55', 'q56',
                                           'q57', 'n_children'], axis = 1)



# renaming columns for further analysis
survey_df.rename(columns={'q1':'age',
                          'q48':'education',
                          'q49':'married',
                          'q54' : 'race',
                          'q55' : 'latino',
                          'q56': 'income',
                          'q57' : 'gender'},
                 inplace=True)



# Subsetting the income groups
survey_df.income.value_counts()
sns.distplot(survey_df['income'])
survey_df.income.describe()
survey_df.income.quantile([0.25,
                           0.50,
                           0.75,])

survey_df['income_1'] = 0
survey_df['income_2'] = 0
survey_df['income_3'] = 0
survey_df['income_4'] = 0

survey_df['income_1'][survey_df['income'] == 1] = 1
survey_df['income_1'][survey_df['income'] == 2] = 2
survey_df['income_1'][survey_df['income'] == 3] = 3
survey_df['income_1'][survey_df['income'] == 4] = 4

survey_df['income_2'][survey_df['income'] == 5] = 5
survey_df['income_2'][survey_df['income'] == 6] = 6
survey_df['income_2'][survey_df['income'] == 7] = 7

survey_df['income_3'][survey_df['income'] == 8] = 8
survey_df['income_3'][survey_df['income'] == 9] = 9
survey_df['income_3'][survey_df['income'] == 10] = 10
survey_df['income_3'][survey_df['income'] == 11] = 11

survey_df['income_4'][survey_df['income'] == 12] = 12
survey_df['income_4'][survey_df['income'] == 13] = 13
survey_df['income_4'][survey_df['income'] == 14] = 14



######################################################
# Step 2: Scale to get equal variance
######################################################

scaler = StandardScaler()


scaler.fit(survey_reduced)


X_scaled_step2 = scaler.transform(survey_reduced)


######################################################
# Step 3: Run PCA without limiting the number of components
######################################################

pca_step3 = PCA(n_components = None,
                           random_state = 508)


pca_step3.fit(X_scaled_step2)


pca_factor_strengths = pca_step3.transform(X_scaled_step2)


######################################################
# Step 4: Analyze the scree plot to determine how many components to retain
######################################################

fig, ax = plt.subplots(figsize=(15, 15))

features = range(pca_step3.n_components_)

plt.plot(features,
       pca_step3.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Reduced Mobile app Survey Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()


######################################################
# Step 5: Run PCA again based on the desired number of components
######################################################

survey_pca_reduced = PCA(n_components = 5,
                           random_state = 508)


survey_pca_reduced.fit(X_scaled_step2)

######################################################
# Step 6: Analyze factor loadings to understand principal components
#####################################################

factor_loadings_df = pd.DataFrame(pd.np.transpose(survey_pca_reduced.components_))


factor_loadings_df = factor_loadings_df.set_index(survey_reduced.columns)


print(factor_loadings_df)


factor_loadings_df.to_excel('practice_factor_loadings.xlsx')

#####################################################
# Step 7: Analyze factor strengths per customer
#####################################################

#X_pca_reduced = pd.read_excel(XXXXXX)


X_pca_reduced = survey_pca_reduced.transform(X_scaled_step2)


X_pca_df = pd.DataFrame(X_pca_reduced)

X_pca_df.columns = ['reserved_latecomers', 'social_network_active',
                    'old_school_tech', 'executive_ceo', 'not_tech_savv']




###############################################################################
# Combining Cluster with PCA
###############################################################################


#####################################################
# Step 1: Scale to get equal variance
#####################################################

scaler = StandardScaler()


scaler.fit(X_pca_df)


X_pca_clust = scaler.transform(X_pca_df)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))


X_pca_clust_df.columns = X_pca_df.columns


########################
# Step 2: Experiment with different numbers of clusters
########################

survey_k_pca = KMeans(n_clusters = 5,
                      random_state = 508)


survey_k_pca.fit(X_pca_clust_df)


survey_kmeans_pca = pd.DataFrame({'cluster': survey_k_pca.labels_})


print(survey_kmeans_pca['cluster'].value_counts())



#####################################################
# Step 3: Analyze cluster memberships
#####################################################

centroids_pca = survey_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['reserved_latecomers', 'social_network_active',
                    'old_school_tech', 'executive_ceo', 'not_tech_savv']


print(centroids_pca_df)


# Sending data to Excel
centroids_pca_df.to_excel('survey_pca_centriods.xlsx')


clst_pca_df = pd.concat([survey_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)

final_pca_clust_df = pd.concat([survey_df.loc[ : , ['age', 'education',
                                                    'married', 'race',
                                                    'latino', 'income',
                                                    'gender', 'n_children',
                                                    'q50r1',
                                           'q50r2', 'q50r3', 'q50r4',
                                           'q50r5','caseID']],
                                clst_pca_df],
                                axis = 1)

final_pca_clust_df.head(5)
data_df = final_pca_clust_df

'''
data_df = pd.concat([survey_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)

X_pca_reduced_df = pd.DataFrame(X_pca_reduced)


X_scaled_reduced_df.columns = customer_features_reduced.columns


clusters_df = pd.concat([customers_kmeans_clusters,
                         X_scaled_reduced_df],
                         axis = 1)

print(clusters_df)
'''

'''

for col in enumerate(survey_df):
    print(col)


df.rename(columns={ df.columns[1]: "whatever" },
                     inplace = True)

survey_df.rename(columns={'q1':'age',
                          'q48':'education',
                          'q49':'married',
                          'q54' : 'race',
                          'q55' : 'latino',
                          'q56': 'income',
                          'q57' : 'gender'},
                 inplace=True)

'''



print(data_df)


data_df.to_excel('data_final.xlsx')


#####################################################
# Analysing the demographics
#####################################################

########################
# Education
########################

# the boxplot is going to increase or decrease the area according to the cluster

# social_network_active
#put the demographic data here and compare with the clusters
fig, ax = plt.subplots(figsize = (12, 10))
sns.boxplot(x = survey_df['education'],
            y = data_df['social_network_active'],
            hue = data_df['cluster'])

plt.ylim(-3, 5)
plt.tight_layout()
plt.show()

# executive_ceo
fig, ax = plt.subplots(figsize = (12, 10))
sns.boxplot(x = survey_df['education'],
            y =data_df['executive_ceo'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3.5,5)
plt.tight_layout()
plt.show()




################################################
# Age
################################################
# social_network_active
fig, ax = plt.subplots(figsize = (12, 10))
sns.boxplot(x = survey_df['age'],
            y = data_df['social_network_active'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3, 4.5)
plt.tight_layout()
plt.show()

# executive_ceo
fig, ax = plt.subplots(figsize = (12, 10))
sns.boxplot(x = survey_df['age'],
            y = data_df['executive_ceo'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3, 5)
plt.tight_layout()
plt.show()




################################################
# Married
################################################
# social_network_active
fig, ax = plt.subplots(figsize = (12, 10))
sns.boxplot(x = survey_df['married'],
            y = data_df['social_network_active'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3, 4.5)
plt.tight_layout()
plt.show()

# executive_ceo
fig, ax = plt.subplots(figsize = (12, 10))
sns.boxplot(x = survey_df['married'],
            y = data_df['executive_ceo'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3, 5)
plt.tight_layout()
plt.show()




################################################
# Race
################################################
# social_network_active
fig, ax = plt.subplots(figsize = (12, 10))
sns.boxplot(x = survey_df['race'],
            y = data_df['social_network_active'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3, 4.5)
plt.tight_layout()
plt.show()

# executive_ceo
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['race'],
            y = data_df['executive_ceo'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3, 5)
plt.tight_layout()
plt.show()






################################################
# Latino
################################################
# social_network_active
fig, ax = plt.subplots(figsize = (12, 10))
sns.boxplot(x = survey_df['latino'],
            y = data_df['social_network_active'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3, 4)
plt.tight_layout()
plt.show()

# executive_ceo
fig, ax = plt.subplots(figsize = (12, 10))
sns.boxplot(x = survey_df['latino'],
            y = data_df['executive_ceo'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3, 5)
plt.tight_layout()
plt.show()




################################################
# Income
################################################
# pca1
fig, ax = plt.subplots(figsize = (15, 18))
sns.boxplot(x = survey_df['income'],
            y = data_df['social_network_active'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3,4)
plt.tight_layout()
plt.show()

# pca2
fig, ax = plt.subplots(figsize = (12, 10))
sns.boxplot(x = survey_df['income'],
            y = data_df['executive_ceo'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3, 5)
plt.tight_layout()
plt.show()


################################################
# Gender
################################################
# pca1
fig, ax = plt.subplots(figsize = (12, 10))
sns.boxplot(x = survey_df['gender'],
            y = data_df['social_network_active'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3, 4)
plt.tight_layout()
plt.show()

# pca2
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['gender'],
            y = data_df['executive_ceo'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-3, 5)
plt.tight_layout()
plt.show()

'''
################################################
# Children
################################################
# pca1
fig, ax = plt.subplots(figsize = (8, 4))
plt.subplot(2,2,1)
sns.boxplot(x = 'q51r1',
            y = 'xxxxxxx',
            hue = 'cluster',
            data = data_df)

plt.ylim(-1, 8)
plt.tight_layout()

# pca2
fig, ax = plt.subplots(figsize = (8, 4))
plt.subplot(2,2,2)
sns.boxplot(x = 'q51r2',
            y = 'xxxx',
            hue = 'cluster',
            data = data_df)

plt.ylim(-3, 5)
plt.tight_layout()

# pca3
fig, ax = plt.subplots(figsize = (8, 4))
plt.subplot(2,2,3)

sns.boxplot(x = 'q51r3',
            y = 'xxxxxxxxxx',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 3)
plt.tight_layout()

# pca3
fig, ax = plt.subplots(figsize = (8, 4))
plt.subplot(2,2,4)
sns.boxplot(x = 'q51r4',
            y = 'xxxxxxxxxx',
            hue = 'cluster',
            data = data_df)

plt.ylim(-4, 3)
plt.tight_layout()
plt.show()

'''

################################################
# Children
################################################
# pca1
fig, ax = plt.subplots(figsize = (12, 10))
sns.boxplot(x = survey_df['n_children'],
            y = data_df['social_network_active'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-2.5, 4)
plt.tight_layout()
plt.show()

# pca2
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = survey_df['n_children'],
            y = data_df['xxxxxxxxxx'],
            hue = data_df['cluster'],
            data = data_df)

plt.ylim(-4, 3)
plt.tight_layout()
plt.show()


'''
Cluster 2 (green) seems to have an inclination towards being socially
      social_network_connected. The education doesn't have a big impact though.

      In terms of age, there is a small group (age under 18) from cluster 2 that is a outlie
       in web social engagement.
       Being marital status doesn't affect a lot the trend, being married
       stands out just a bit, cluster 2 still stands out
       NAtive HAwaiian or pacific islander (in cluster 1) and American Indian
            or alaska native (in cluster 4) are outliers.

      Latinos in cluster 2(green) tend to have more of this behavior

      income 1 and income 8 (green cluster) were the most active

      gender didn't affect

      Families have this behavior
'''
