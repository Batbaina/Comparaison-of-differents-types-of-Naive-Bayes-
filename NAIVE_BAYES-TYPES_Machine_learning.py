#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[5]:


#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',50)
plt.style.use('default')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score,  roc_auc_score,balanced_accuracy_score,jaccard_score,cohen_kappa_score
from sklearn.metrics import matthews_corrcoef

from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB,ComplementNB


# In[ ]:





# # DATA EXPLORATION ANALYSIS (EDA) 

# In[ ]:





# In[6]:



import sympy
#latex_expr = sympy.latex(namefile)


# In[7]:


#Importing data
odata = pd.read_csv(r"odata.csv")


# In[8]:


#Exploratory Data Analysis
odata.head()


# In[9]:


odata.columns


# In[10]:


odata.tail()


# In[11]:


#Dimension of the dataset
odata.shape


# In[12]:


#The difference proportion  of the classes
odata['overweight'].value_counts()


# In[13]:


odata.isna().sum() #checking for null values


# In[14]:


#The description of the Dataset
odata.describe()


# In[ ]:





# In[ ]:





# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Histograms
odata.hist(figsize=(20, 15))
plt.show()


# In[ ]:





# In[16]:


# Correlation matrix
plt.figure(figsize=(35,30))
corr_matrix = odata.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[53]:


#To check our target variable

import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
ax = odata["overweight"].value_counts(normalize=True).mul(100).plot.bar()
for p in ax.patches:
    y = p.get_height()
    x = p.get_x() + p.get_width() / 2

    # Label of bar height
    label = "{:.1f}%".format(y)

    # Annotate plot
    ax.annotate(label, xy=(x, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=14)

# Remove y axis
ax.get_yaxis().set_visible(False)
plt.savefig('LM5.jpg',bbox_inches='tight', dpi=150)


# In[ ]:





# In[ ]:





# # Splitting

# In[17]:


#Splitting
X = odata.drop('overweight', axis =1 )
y = odata['overweight']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,stratify=y,test_size = 0.2)
print(f'X_train shape : {X_train.shape}\nX_Test shape: {X_test.shape} \ny_train shape : {y_train.shape}\ny_test shape: {y_test.shape}')
y_train.value_counts()


# # The min max scaler

# In[18]:


#Most of the Naive bayes algorithm do not accept negative values for training,
#therefore it is not ideal to use a standard scaler which outputs values between -1 to 1. 
#Then best thing to do is to use the min max scaler with range 0 to 1

scaler=MinMaxScaler(feature_range = (0,1))
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# # Creating dataframe to store model results

# In[19]:


#creating dataframe to store model results
model_results = pd.DataFrame(columns=["Model", "Accuracy Score"])
#Testing various classifiers to see which gives the best accuracy score
models = [
("Gaussian Naive Bayes", GaussianNB()),
("Bernoulli Naive Bayes", BernoulliNB()),
('Multinomial Naive Bayes', MultinomialNB()),
('Complement Naive Bayes', ComplementNB()),
('Categorical Naive Bayes', CategoricalNB())
]


# Define the models without the name to be use for a ROC CURVE
models_0 = [ GaussianNB(),BernoulliNB(),MultinomialNB(),ComplementNB(),CategoricalNB()]

for clf_name, clf in tqdm(models):
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    score = accuracy_score(y_test, predictions)
    ypred_prob = clf.predict_proba(X_test)[:, 1]
    rocAuc_score = roc_auc_score(y_test, ypred_prob)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    bal_score = balanced_accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)
    cks = cohen_kappa_score(y_test, predictions)
    jac = jaccard_score(y_test, predictions)
    new_row = {"Model": clf_name, "Accuracy Score": score, 'jaccard score':jac, 'kappa_score':cks,'balanced_acc_score' : bal_score, 'Roc_Auc_score':rocAuc_score,'precision':precision,'recall':recall, 'mcc' : mcc,'f1_score':f1}
    model_results = model_results.append(new_row, ignore_index=True)


# In[ ]:





# # Confusion Matrix 

# In[20]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create a subplot to display multiple confusion matrices
num_models = len(models)
num_cols = 2  # Number of columns in the subplot grid
num_rows = (num_models + 1) // num_cols  # Number of rows in the subplot grid

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

for i, (model_name, model) in enumerate(models):
    row = i // num_cols
    col = i % num_cols

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    ax = axes[row, col] if num_rows > 1 else axes[col]
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_title(f"Confusion Matrix for {model_name}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    # Print the confusion matrix for each model
    #print(f"Confusion Matrix for {model_name}:")
    #print(cm)
    #print()

# Remove empty subplots if the number of models is not a perfect square
if num_models % num_cols != 0:
    for i in range(num_models, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

plt.tight_layout()
plt.show()


# In[ ]:





# In[21]:


#Results 
model_results.sort_values(by="Accuracy Score", ascending=False)

#Defining custom function which returns the list for df.style.apply() method
def highlight_max(s):
    if s.dtype == np.object:
        is_max = [False for _ in range(s.shape[0])]
    else:
        is_max = s == s.max()
    return ['background: lightgreen' if cell else '' for cell in is_max]

def highlight_min(s):
    if s.dtype == np.object:
        is_min = [False for _ in range(s.shape[0])]
    else:
        is_min = s == s.min()
    return ['background: red' if cell else '' for cell in is_min]

model_results.style.apply(highlight_max)
model_results.style.apply(highlight_min)
model_results


# In[ ]:





# # Visualization of results

# In[29]:



#Visualization of results
plt.figure(figsize = (15,4))
sns.barplot(model_results['Model'],model_results['Accuracy Score'],order = model_results.sort_values(by = 'Accuracy Score',ascending = False)['Model'])
plt.title('Barplot of Accuracy Score for models');

plt.figure(figsize = (15,4))
sns.barplot(model_results['Model'],model_results['Roc_Auc_score'], order = model_results.sort_values(by = 'Roc_Auc_score',ascending = False)['Model'])
plt.title('Barplot of Roc_Auc_score for models');

plt.figure(figsize = (15,4))
sns.barplot(model_results['Model'],model_results['precision'], order = model_results.sort_values(by = 'precision',ascending = False)['Model'])
plt.title('Barplot of Precision for models');

plt.figure(figsize = (15,4))
sns.barplot(model_results['Model'],model_results['recall'], order = model_results.sort_values(by = 'recall',ascending = False)['Model'])
plt.title('Barplot of Recall for models');


# In[30]:


plt.figure(figsize = (13,6))
plt.plot(model_results['Model'],model_results['jaccard score'],'g--o', label = 'Jaccard score')
plt.plot(model_results['Model'],model_results['precision'],'b--o',label = 'Precision')
plt.plot(model_results['Model'],model_results['recall'],'k--o',label = 'Recall')
plt.legend(loc =[0.44,0.4])
plt.xlabel('Models')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


# In[31]:


plt.figure(figsize = (13,7))
plt.plot(model_results['Model'],model_results['Accuracy Score'],'g--o', label = 'Accuracy Score')
plt.plot(model_results['Model'],model_results['balanced_acc_score'],'b--o',label = 'balanced_acc_score')
plt.plot(model_results['Model'],model_results['Roc_Auc_score'],'r--o',label = 'Roc_Auc_score')
plt.legend(loc =[1.02,0.85])
plt.xlabel('Models')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


# In[32]:


plt.figure(figsize = (13,6))
plt.plot(model_results['Model'],model_results['f1_score'],'g--o', label = 'f1_score')
plt.plot(model_results['Model'],model_results['mcc'],'b--o',label = 'mcc')
plt.plot(model_results['Model'],model_results['jaccard score'],'c--o',label = 'jaccard score')
plt.legend(loc =[0.4,0.4])
plt.xlabel('Models')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


# # ROC CURVE

# In[33]:


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

# Train the models and record the results
for cls in models_0:
    model = cls.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    result_table = result_table.append({'classifiers': cls.__class__.__name__,
                                        'fpr': fpr,
                                        'tpr': tpr,
                                        'auc': auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(15, 10))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], result_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0, 1], [0, 1], color='yellow', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 8}, loc='lower right')

plt.show()

# Print the result table
#print(result_table)


# # Handling Data imbalance

# In[59]:



#Handling Data imbalance
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

model_results_resampled = pd.DataFrame(columns=["Model", "Resampled Accuracy Score"])

for clf_name, clf in tqdm(models):
    clf.fit(X_res, y_res)
    predictions = clf.predict(X_test)
    score = accuracy_score(y_test, predictions)
    ypred_prob = clf.predict_proba(X_test)[:, 1]
    rocAuc_score = roc_auc_score(y_test, ypred_prob)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    bal_score = balanced_accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)
    cks = cohen_kappa_score(y_test, predictions)
    jac = jaccard_score(y_test, predictions)
    resampled_result = {"Model": clf_name, "Resampled Accuracy Score": score, 'Resampled Jaccard Score':jac, 'Resampled kappa_score':cks,'Resampled Balanced_acc_score' : bal_score, 'Resampled Roc_Auc_score':rocAuc_score,'Resampled precision':precision,'Resampled recall':recall, 'Resampled mcc' : mcc,'Resampled f1_score':f1}


# In[60]:


#To check our target variable

import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
ax =y_res.value_counts(normalize=True).mul(100).plot.bar()
for p in ax.patches:
    y = p.get_height()
    x = p.get_x() + p.get_width() / 2

    # Label of bar height
    label = "{:.1f}%".format(y)

    # Annotate plot
    ax.annotate(label, xy=(x, y), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=14)

# Remove y axis
ax.get_yaxis().set_visible(False)
plt.savefig('LM5.jpg',bbox_inches='tight', dpi=150)


# # Confusion Matrix 

# In[35]:


# Create a subplot to display multiple confusion matrices
num_models = len(models)
num_cols = 2  # Number of columns in the subplot grid
num_rows = (num_models + 1) // num_cols  # Number of rows in the subplot grid

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

for i, (model_name, model) in enumerate(models):
    row = i // num_cols
    col = i % num_cols

    model.fit(X_res, y_res)
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    ax = axes[row, col] if num_rows > 1 else axes[col]
    sns.heatmap(cm, annot=True, ax=ax)
    ax.set_title(f"Confusion Matrix for {model_name}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    # Print the confusion matrix for each model
    #print(f"Confusion Matrix for {model_name}:")
    #print(cm)
    #print()

# Remove empty subplots if the number of models is not a perfect square
if num_models % num_cols != 0:
    for i in range(num_models, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

plt.tight_layout()
plt.show()


# In[36]:


model_results_resampled = model_results_resampled.append(resampled_result,ignore_index=True)
model_results_resampled.sort_values(by="Resampled Accuracy Score", ascending=False)
new_model_results = model_results.set_index('Model').transpose()
new_model_results


# In[37]:


plt.figure(figsize = (13,6))
plt.plot(model_results_resampled['Model'],model_results_resampled['Resampled Jaccard Score'],'g--o', label = 'Jaccard score')
plt.plot(model_results_resampled['Model'],model_results_resampled['Resampled precision'],'b--o',label = 'Precision')
plt.plot(model_results_resampled['Model'],model_results_resampled['Resampled recall'],'k--o',label = 'Recall')
plt.legend(loc =[0.44,0.4])
plt.xlabel('Models')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


# In[38]:


plt.figure(figsize = (13,6))
plt.plot(model_results_resampled['Model'],model_results_resampled['Resampled f1_score'],'g--o', label = 'f1_score')
plt.plot(model_results_resampled['Model'],model_results_resampled['Resampled mcc'],'b--o',label = 'mcc')
plt.plot(model_results_resampled['Model'],model_results_resampled['Resampled Jaccard Score'],'c--o',label = 'jaccard score')
plt.legend()
plt.xlabel('Models')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


# In[39]:


plt.figure(figsize = (13,7))
plt.plot(model_results_resampled['Model'],model_results_resampled['Resampled Accuracy Score'],'g--o', label = 'Accuracy Score')
plt.plot(model_results_resampled['Model'],model_results_resampled['Resampled Balanced_acc_score'],'b--o',label = 'balanced_acc_score')
plt.plot(model_results_resampled['Model'],model_results_resampled['Resampled Roc_Auc_score'],'r--o',label = 'Roc_Auc_score')
plt.legend()
plt.xlabel('Models')
plt.ylabel('Value')
plt.tight_layout()
plt.show()


# In[40]:


plt.style.use('ggplot')
ax = new_model_results.plot(kind='bar', figsize=(15, 8), rot=0, colormap = 'rainbow')
ax.legend()
# plt.suptitle("Comparison of Perofrmace metrics for all Naive Bayes Algorithms ", size =15)
plt.savefig('Comparison_of_performance_for_all_Bayes.jpg',bbox_inches='tight')
plt.show()


# In[41]:


new_model_results_resampled = model_results_resampled.set_index('Model').transpose()
ax = new_model_results_resampled.plot(kind='bar', figsize=(15, 8), rot=90,colormap = 'rainbow')
ax.legend(loc = [0.6,0.8])
plt.show()


# In[42]:


df = pd.merge(model_results,model_results_resampled,on = 'Model').set_index('Model')
ax = df[['mcc','Resampled mcc']].plot(kind='bar', figsize=(15, 6), rot=0)
ax.legend()
plt.show()


# # ROC CURVE AFTER RESAMPLING

# In[62]:


# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

# Train the models and record the results
for cls in models_0:
    model = cls.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    result_table = result_table.append({'classifiers': cls.__class__.__name__,
                                        'fpr': fpr,
                                        'tpr': tpr,
                                        'auc': auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(15, 10))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], result_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0, 1], [0, 1], color='yellow', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 8}, loc='lower right')

plt.show()

# Print the result table
#print(result_table)


# From the analysis, we can observe the following:
# 		
# 		Bernoulli Naive Bayes and Categorical Naive Bayes models have the highest accuracy scores of 0.785185.
# 		Gaussian Naive Bayes has the highest $Roc\_Auc_score$ of 0.663305.
# 		Complement Naive Bayes has the highest balanced\_acc\_score of 0.643624.
# 		Gaussian Naive Bayes has the highest F1\_score of 0.366197.
# 		Complement Naive Bayes has the highest Jaccard score of 0.279412.
# 		Complement Naive Bayes has the highest Kappa\_score of 0.210715.
# 		Bernoulli Naive Bayes has the highest MCC (Matthews Correlation Coefficient) of 0.236200.
# 		Bernoulli Naive Bayes has the highest precision of 0.500000.
# 		Complement Naive Bayes has the highest recall of 0.655172.\\
# 		
# 		Based on these findings, the choice of the best Naive Bayes model depends on the specific evaluation metric that is most important for the classification task at hand. For example, if accuracy is the primary concern, Bernoulli Naive Bayes or Categorical Naive Bayes may be the best choices. On the other hand, if ROC AUC score or balanced accuracy score is more important, Gaussian Naive Bayes or Complement Naive Bayes may be preferred.
# 		
# 		From these results, it can be observed that Bernoulli Naive Bayes and Categorical Naive Bayes models have the highest accuracy scores of 0.785185. However, it is important to consider other evaluation metrics as well to get a comprehensive understanding of the model's performance. Then when we look at the ROC curve, base on the AUC values, the best performing Naive Bayes classifier among the ones evaluated is CategoricalNB, with an AUC of 0.664118.
# 		

# # Models comparaison

# In[44]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[45]:


models_1 = [("Logistic Regression", LogisticRegression(random_state=101)),

('Categorical Naive Bayes', CategoricalNB()),
("LightGBM", LGBMClassifier(random_state=101)),
#("XGBoost", XGBClassifier(random_state=101)),
#("CatBoost", CatBoostClassifier(verbose = False,random_state = 101,)),
("KNN", KNeighborsClassifier()),
("Random Forest", RandomForestClassifier()),
("LDA", LinearDiscriminantAnalysis())
]


# In[46]:


model_results_com = pd.DataFrame(columns=["Model", "Accuracy Score"])


# In[47]:




for clf_name, clf in tqdm(models_1):
    clf.fit(X_res, y_res)
    predictions = clf.predict(X_test)
    score = accuracy_score(y_test, predictions)
    ypred_prob = clf.predict_proba(X_test)[:, 1]
    rocAuc_score = roc_auc_score(y_test, ypred_prob)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    bal_score = balanced_accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)
    cks = cohen_kappa_score(y_test, predictions)
    jac = jaccard_score(y_test, predictions)
    new_row = {"Model": clf_name, "Accuracy Score": score, 'jaccard score':jac, 'kappa_score':cks,'balanced_acc_score' : bal_score, 'Roc_Auc_score':rocAuc_score,'precision':precision,'recall':recall, 'mcc' : mcc,'f1_score':f1}
    model_results_com = model_results_com.append(new_row, ignore_index=True)


# In[48]:


#Results 
#model_results_com.sort_values(by="Accuracy Score", ascending=False)


# In[49]:


#Results 
model_results_com.sort_values(by="Accuracy Score", ascending=False)

#Defining custom function which returns the list for df.style.apply() method
def highlight_max(s):
    if s.dtype == np.object:
        is_max = [False for _ in range(s.shape[0])]
    else:
        is_max = s == s.max()
    return ['background: lightgreen' if cell else '' for cell in is_max]

def highlight_min(s):
    if s.dtype == np.object:
        is_min = [False for _ in range(s.shape[0])]
    else:
        is_min = s == s.min()
    return ['background: red' if cell else '' for cell in is_min]

model_results_com.style.apply(highlight_max)
model_results_com.style.apply(highlight_min)


# In[ ]:





# In[ ]:





# In[50]:


models_1 = [LogisticRegression(random_state=101), CategoricalNB(),
LGBMClassifier(random_state=101),
#("XGBoost", XGBClassifier(random_state=101)),
#("CatBoost", CatBoostClassifier(verbose = False,random_state = 101,)),
 KNeighborsClassifier(),
RandomForestClassifier(),
LinearDiscriminantAnalysis()
]


# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

# Train the models and record the results
for cls in models_1:
    model = cls.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    result_table = result_table.append({'classifiers': cls.__class__.__name__,
                                        'fpr': fpr,
                                        'tpr': tpr,
                                        'auc': auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(15, 10))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], result_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0, 1], [0, 1], color='yellow', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 8}, loc='lower right')

plt.show()

# Print the result table
#print(result_table)


#  Based on these results, the Random Forest algorithm achieved the highest accuracy score of 0.762963, followed by LightGBM with an accuracy score of 0.718519. However, it's important to consider other evaluation metrics as well, such as F1 score, precision, recall, and ROC AUC score, to make a comprehensive assessment of the algorithms' performance.\\
# 
# Based on the F1-Score, which is a commonly used metric that balances precision and recall, the CategoricalNB algorithm achieved the highest score of 0.664118. However, it's important to consider other factors such as the specific requirements of the problem, the interpretability of the algorithm, and the computational complexity.
# 
# It is recommended to evaluate the algorithms based on multiple metrics and consider the specific needs of the problem to determine the best algorithm for the given scenario.

# In[ ]:





# In[ ]:





# In[ ]:





# # CONCLUSION ON THE CODE

# 	Based on these findings, the choice of the best Naive Bayes model depends on the specific evaluation metric that is most important for the classification task at hand. For example, if accuracy is the primary concern, Bernoulli Naive Bayes or Categorical Naive Bayes may be the best choices. On the other hand, if ROC AUC score or balanced accuracy score is more important, Gaussian Naive Bayes or Complement Naive Bayes may be preferred. But on this kind of categorical the Categorical Naive Bayes is the one that have a good performance.
# 	
# 	It is important to consider multiple evaluation metrics to get a comprehensive understanding of the model's performance. Additionally, other factors such as the specific requirements of the problem, interpretability of the algorithm, and computational complexity should also be taken into account when selecting the best algorithm for a given scenario.




