#!/usr/bin/env python
# coding: utf-8

# <div style="color:black;background-color:brown;padding:3%;border-radius:150px 150px;font-size:2em;text-align:center">Logistic H1N1 Prediction
# </div>

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


# In[7]:


vaccine=pd.read_csv('h1n1_vaccine_prediction.csv')
vaccine.head()


# In[8]:


vaccine.info()


# In[10]:


vaccine.shape


# In[11]:


vaccine.dtypes.value_counts()


# In[12]:


vaccine.columns


# In[13]:


vaccine.drop(['unique_id'], inplace = True, axis = 1)


# In[14]:


vaccine.isnull().sum()


# In[15]:


vaccine['h1n1_worry'].unique()
# 0=Not worried at all, 1=Not very worried, 2=Somewhat worried, 3=Very worried


# In[16]:


vaccine['h1n1_worry'].value_counts().plot(kind = 'bar')
plt.xlabel('H1N1 Worry')
plt.ylabel('Count')
plt.show()


# In[17]:


display(vaccine['h1n1_worry'].mode(), vaccine['h1n1_worry'].median())


# In[18]:


# Filling 92 missing values by mode
vaccine['h1n1_worry'].fillna(vaccine['h1n1_worry'].mode()[0], inplace = True)


# In[19]:


vaccine['h1n1_awareness'].unique()
# 0=No knowledge, 1=little knowledge, 2=good knowledge


# In[20]:


sns.countplot(x = 'h1n1_awareness', data = vaccine, hue = 'h1n1_vaccine')
plt.show()


# In[21]:


vaccine['h1n1_awareness'].mode()


# In[22]:


# Filling 192 missing values by mode
vaccine['h1n1_awareness'].fillna(vaccine['h1n1_awareness'].mode()[0], inplace = True)


# In[23]:


vaccine['antiviral_medication'].unique()
# 0=no, 1=yes


# In[24]:


vaccine['antiviral_medication'].value_counts().plot(kind = 'pie', autopct = '%0.2f%%', figsize = [5,5], explode = [0,0.2])
plt.show()


# In[25]:


display(vaccine['antiviral_medication'].mode())
# missing 71 missing values by mode
vaccine['antiviral_medication'].fillna(vaccine['antiviral_medication'].mode()[0], inplace = True)


# In[26]:


vaccine['contact_avoidance'].unique()
# 0=no, 1=yes


# In[27]:


vaccine['contact_avoidance'].value_counts()


# In[28]:


sns.countplot(y = 'contact_avoidance', data = vaccine, color = 'green')
plt.show()


# In[29]:


# filling 208 missing values by mode
vaccine['contact_avoidance'].fillna(vaccine['contact_avoidance'].mode()[0], inplace = True)


# In[30]:


vaccine['bought_face_mask'].unique()
# 0=no, 1=yes


# In[31]:


vaccine['bought_face_mask'].value_counts().plot(kind = 'bar', color = 'orange')
plt.xlabel('bought_face_mask')
plt.ylabel('count')
plt.show()


# In[32]:


# filling 19 missing values by mode
vaccine['bought_face_mask'].fillna(vaccine['bought_face_mask'].mode()[0], inplace = True)


# In[33]:


vaccine['wash_hands_frequently'].unique()
# 0-Washes hands frequently, 1=uses hand sanitizer -


# In[34]:


sns.countplot(x = 'wash_hands_frequently', data = vaccine)
plt.show()


# In[35]:


# filling 42 missing values by mode
vaccine['wash_hands_frequently'].fillna(vaccine['wash_hands_frequently'].mode()[0], inplace = True)


# In[36]:


vaccine['avoid_large_gatherings'].unique()
# 0=no, 1=yes


# In[37]:


vaccine['avoid_large_gatherings'].value_counts()


# In[38]:


# filling 87 missing values by mode
vaccine['avoid_large_gatherings'].fillna(vaccine['avoid_large_gatherings'].mode()[0], inplace = True)


# In[39]:


vaccine['reduced_outside_home_cont'].unique()
# 0=no, 1=yes


# In[40]:


vaccine['reduced_outside_home_cont'].value_counts()


# In[41]:


# filling 82 missing values by mode
vaccine['reduced_outside_home_cont'].fillna(vaccine['reduced_outside_home_cont'].mode()[0], inplace = True)


# In[42]:


vaccine['avoid_touch_face'].unique()
# 0=no, 1=yes


# In[43]:


vaccine['avoid_touch_face'].value_counts()


# In[44]:


# filling 128 missing values by mode
vaccine['avoid_touch_face'].fillna(vaccine['avoid_touch_face'].mode()[0], inplace = True)


# In[45]:


display(vaccine['dr_recc_h1n1_vacc'].unique(), vaccine['dr_recc_seasonal_vacc'].unique())
# 0=no, 1=yes


# In[46]:


fig, ax = plt.subplots(1,2, figsize = [7,6], sharey = True )
sns.countplot(x = 'dr_recc_h1n1_vacc', data = vaccine, color = 'maroon', ax=ax[0])
sns.countplot(x = 'dr_recc_seasonal_vacc', data = vaccine, color = 'navy', ax=ax[1])
plt.show()


# In[47]:


# filling 2160 missing values by mode
vaccine['dr_recc_h1n1_vacc'].fillna(vaccine['dr_recc_h1n1_vacc'].mode()[0], inplace = True)
vaccine['dr_recc_seasonal_vacc'].fillna(vaccine['dr_recc_seasonal_vacc'].mode()[0], inplace = True)


# In[48]:


vaccine['chronic_medic_condition'].value_counts()


# In[49]:


# filling 971 missing values by mode
vaccine['chronic_medic_condition'].fillna(vaccine['chronic_medic_condition'].mode()[0], inplace = True)


# In[50]:


vaccine['cont_child_undr_6_mnths'].value_counts().plot(kind = 'barh', cmap = 'rainbow', edgecolor = 'b')


# In[51]:


# filling 820 missing values by mode
vaccine['cont_child_undr_6_mnths'].fillna(vaccine['cont_child_undr_6_mnths'].mode()[0], inplace = True)


# In[52]:


vaccine['is_health_worker'].value_counts()


# In[53]:


# filling 804 missing values by mode
vaccine['is_health_worker'].fillna(vaccine['is_health_worker'].mode()[0], inplace = True)


# In[54]:


vaccine['has_health_insur'].unique()
# 0=no, 1=yes


# In[55]:


vaccine['has_health_insur'].value_counts().plot(kind = 'barh')


# In[56]:


vaccine['has_health_insur'].fillna(2.0, inplace = True)


# In[57]:


vaccine['has_health_insur'].value_counts().plot(kind = 'barh', color = 'b', edgecolor = 'r')


# In[58]:


display(vaccine['is_h1n1_vacc_effective'].unique(), vaccine['is_seas_vacc_effective'].unique())
# 1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 
# 4=Thinks it is somewhat effective, 5=Thinks it is highly effective


# In[59]:


colors = ['#CAFF70', '#FF1493', '#00BFFF', '#FFD700', '#836FFF']
colors1 = ['#FF7F24', '#FFB90F', '#A2CD5A', '#BF3EFF', '#EEAEEE']
fig, (ax1,ax2) = plt.subplots(1,2, figsize = [10,10])

ax1.pie(vaccine['is_h1n1_vacc_effective'].value_counts(), labels = vaccine['is_h1n1_vacc_effective'].value_counts().index , 
        autopct = '%0.2f%%', explode= [0.1,0,0,0,0], colors = colors, shadow = True)

ax2.pie(vaccine['is_seas_vacc_effective'].value_counts(), labels = vaccine['is_seas_vacc_effective'].value_counts().index , 
        autopct = '%0.2f%%', explode= [0.1,0,0,0,0], colors = colors1, shadow = True)

ax1.set_title('is_h1n1_vacc_effective')
ax2.set_title('is_seas_vacc_effective')

plt.show()


# In[60]:


# filling 391 and 462 missing values respectively by mode
vaccine['is_h1n1_vacc_effective'].fillna(vaccine['is_h1n1_vacc_effective'].mode()[0], inplace = True)
vaccine['is_seas_vacc_effective'].fillna(vaccine['is_seas_vacc_effective'].mode()[0], inplace = True)


# In[61]:


display(vaccine['is_h1n1_risky'].unique(), vaccine['is_seas_risky'].unique())
# 1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=don’t know if it is risky or not, 
# 4=Thinks it is a somewhat high risk, 5=Thinks it is very highly risky


# In[62]:


fig, ax = plt.subplots(1, 2, figsize = [10,10])
vaccine['is_h1n1_risky'].value_counts().plot(kind = 'pie', autopct = '%0.2f%%', explode = [0.05,0,0,0,0], cmap = 'RdYlGn', ax = ax[0])
vaccine['is_seas_risky'].value_counts().plot(kind = 'pie', autopct = '%0.2f%%', explode = [0.05,0,0,0,0], cmap = 'Paired', ax = ax[1])
plt.show()


# In[63]:


# filling 388 and 514 missing values respectively by mode
vaccine['is_h1n1_risky'].fillna(vaccine['is_h1n1_risky'].mode()[0], inplace = True)
vaccine['is_seas_risky'].fillna(vaccine['is_seas_risky'].mode()[0], inplace = True)


# In[64]:


display(vaccine['sick_from_h1n1_vacc'].unique(), vaccine['sick_from_seas_vacc'].unique())
# 1=Respondent not worried at all, 2=Respondent is not very worried, 3=Doesn't know, 4=Respondent is somewhat worried, 
# 5Respondent is very worried


# In[65]:


fig, ax = plt.subplots(1,2, figsize = [10,10])
vaccine['sick_from_h1n1_vacc'].value_counts().plot(kind = 'pie', autopct = '%0.2f%%', explode = [0.05,0,0,0,0], cmap = 'Spectral', ax = ax[0])
vaccine['sick_from_seas_vacc'].value_counts().plot(kind = 'pie', autopct = '%0.2f%%', explode = [0.05,0,0,0,0], cmap = 'twilight', ax = ax[1])
plt.show()


# In[66]:


# filling 395 and 537 missing values respectively by mode
vaccine['sick_from_h1n1_vacc'].fillna(vaccine['sick_from_h1n1_vacc'].mode()[0], inplace = True)
vaccine['sick_from_seas_vacc'].fillna(vaccine['sick_from_seas_vacc'].mode()[0], inplace = True)


# In[67]:


vaccine.groupby(['dr_recc_seasonal_vacc']).agg({'is_seas_vacc_effective' : ['count'],
                                      'is_seas_risky' :['count'],
                                      'sick_from_seas_vacc' : ['count']}).plot(kind = 'bar', cmap = 'tab20b', figsize = [5,5])
plt.ylabel('count')
plt.show()


# In[68]:


vaccine.groupby(['dr_recc_h1n1_vacc']).agg({'is_h1n1_vacc_effective' : ['count'],
                                      'is_h1n1_risky' :['count'],
                                      'sick_from_h1n1_vacc' : ['count']}).plot(kind = 'bar', cmap = 'tab20c', figsize = [5,5])
plt.ylabel('count')
plt.show()


# In[69]:


vaccine['qualification'].unique()


# In[70]:


vaccine['qualification'].value_counts()


# In[71]:


# filling 1407 missing values by mode
vaccine['qualification'].fillna(vaccine['qualification'].mode()[0], inplace = True)


# In[72]:


vaccine['sex'].value_counts().plot(kind = 'barh', color = 'c', edgecolor = 'r')
# no nan values


# In[73]:


vaccine['income_level'].unique()


# In[74]:


vaccine['income_level'].value_counts().plot(kind = 'pie', autopct = '%0.2f%%', cmap = 'magma',
                                            explode = [0,0.1,0], figsize = [5,5], shadow = True)


# In[75]:


# no of missing values is 4423, better to create a new category as 'Unknown' as this will not skew the data and info will not be lost
vaccine['income_level'].fillna('Unknown', inplace = True)


# In[76]:


vaccine['marital_status'].unique()


# In[77]:


vaccine['marital_status'].value_counts()


# In[78]:


# filling 1408 missing values by mode
vaccine['marital_status'].fillna(vaccine['marital_status'].mode()[0], inplace = True)


# In[79]:


vaccine['housing_status'].unique()


# In[80]:


vaccine['housing_status'].value_counts().plot(kind = 'bar', color = 'b', edgecolor = 'r')


# In[81]:


# filling 2402 missing values by mode
vaccine['housing_status'].fillna(vaccine['housing_status'].mode()[0], inplace = True)


# In[82]:


vaccine['employment'].unique()


# In[83]:


vaccine['employment'].value_counts().plot(kind = 'pie', autopct = '%0.2f%%', figsize = [5,5])


# In[84]:


# filling 1463 missing values by mode
vaccine['employment'].fillna(vaccine['employment'].mode()[0], inplace = True)


# In[85]:


display(vaccine['no_of_adults'].unique(), vaccine['no_of_children'].unique())


# In[86]:


fig, ax = plt.subplots(1,2, figsize = [7,5], sharey = True)
sns.countplot( x = 'no_of_adults', data = vaccine, ax = ax[0])
sns.countplot( x = 'no_of_children', data = vaccine, ax = ax[1])
plt.show()


# In[87]:


# filling 249 nan values by mode
vaccine['no_of_adults'].fillna(vaccine['no_of_adults'].mode()[0], inplace = True)
vaccine['no_of_children'].fillna(vaccine['no_of_children'].mode()[0], inplace = True)


# In[88]:


vaccine['age_bracket'].value_counts()


# In[89]:


vaccine.isnull().sum()


# In[90]:


vaccine1 = vaccine.astype({'h1n1_worry' : str, 'h1n1_awareness': str, 'antiviral_medication': str,
       'contact_avoidance': str, 'bought_face_mask': str, 'wash_hands_frequently': str,
       'avoid_large_gatherings': str, 'reduced_outside_home_cont': str,
       'avoid_touch_face': str, 'dr_recc_h1n1_vacc': str, 'dr_recc_seasonal_vacc': str,
       'chronic_medic_condition': str, 'cont_child_undr_6_mnths': str,
       'is_health_worker': str, 'has_health_insur': str, 'is_h1n1_vacc_effective': str,
       'is_h1n1_risky': str, 'sick_from_h1n1_vacc': str, 'is_seas_vacc_effective': str, 'age_bracket' : str,
       'is_seas_risky': str, 'sick_from_seas_vacc': str, 'no_of_adults' : str, 'no_of_children' : str,
        'h1n1_vaccine' : str})
vaccine1.dtypes.value_counts()


# In[91]:


vaccine_dummy = pd.get_dummies(vaccine1, drop_first= True )


# In[92]:


vaccine_dummy.shape


# In[93]:


vaccine_dummy['h1n1_vaccine_1'].value_counts(normalize = True)


# In[94]:


y=vaccine_dummy['h1n1_vaccine_1']
x=vaccine_dummy.drop(['h1n1_vaccine_1'], axis=1)


# In[95]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data['Var'] = x.columns
vif_data['VIF'] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]


# In[96]:


vif_data.sort_values(by = ['VIF'], ascending = False)


# In[97]:


x_train, x_test, y_train , y_test=train_test_split(x,y, test_size=.25, random_state=88)


# In[98]:


log = LogisticRegression()

log.fit(x_train,y_train)

print(f"Training Score : {log.score(x_train, y_train)}")
print(f"Testing Score : {log.score(x_test, y_test)}")


# In[99]:


pred_train=log.predict(x_train)
pred_test=log.predict(x_test)


# In[100]:


cnfTrain=pd.DataFrame(metrics.confusion_matrix(y_train , pred_train), columns=["Pred_0", "Pred_1"],
                index=["Act_0", "Act_1"])
cnfTrain


# In[101]:


print(metrics.classification_report(y_train, pred_train))


# In[102]:


cnfTest=pd.DataFrame(metrics.confusion_matrix(y_test , pred_test), columns=["Pred_0", "Pred_1"],
                index=["Act_0", "Act_1"])
cnfTest


# In[103]:


print(metrics.classification_report(y_test, pred_test))


# In[104]:


pd.DataFrame(log.predict_proba(x_train), columns=["Prob_0", "Prob_1"])


# In[105]:


x_train1=x_train.copy()

x_train1["Actual_Default"]=y_train
x_train1["Prob_Default"]=log.predict_proba(x_train)[:, 1]

x_train1


# In[106]:


# Decile Analysis
def profile_decile(X,y,trained_model):
    X_1=X.copy()
    y_1=y.copy()
    y_pred1=trained_model.predict(X_1)
    X_1["Prob_Event"]=trained_model.predict_proba(X_1)[:,1]
    X_1["Y_actual"]=y_1
    X_1["Y_pred"]=y_pred1
    X_1["Rank"]=pd.qcut(X_1["Prob_Event"], 10, labels=np.arange(0,10,1))
    X_1["numb"]=10
    X_1["Decile"]=X_1["numb"]-X_1["Rank"].astype("int")
    
    profile=pd.DataFrame(X_1.groupby("Decile")                         .apply(lambda x: pd.Series({
        'min_score'   : x["Prob_Event"].min(),
        'max_score'   : x["Prob_Event"].max(),
        'Event'       : x["Y_actual"].sum(),
        'Non_event'   : x["Y_actual"].count()-x["Y_actual"].sum(),
        'Total'       : x["Y_actual"].count() })))
    return profile


# In[107]:


newtrain_pred=np.where(log.predict_proba(x_train)[:,1] > 0.232280104, 1, 0) # based on KS value


# In[108]:


cnfNewTrain=pd.DataFrame(metrics.confusion_matrix(y_train , newtrain_pred), columns=["Pred_0", "Pred_1"],
                index=["Act_0", "Act_1"])
cnfNewTrain


# In[109]:


print(metrics.classification_report(y_train , newtrain_pred))


# In[110]:


newtest_pred=np.where(log.predict_proba(x_test)[:,1] > 0.229209326, 1, 0) # based on KS value


# In[111]:


cnfNewTest=pd.DataFrame(metrics.confusion_matrix(y_test , newtest_pred), columns=["Pred_0", "Pred_1"],
                index=["Act_0", "Act_1"])
cnfNewTest


# In[112]:


print(metrics.classification_report(y_test , newtest_pred))


# In[113]:


# Receiver Operating Characterestics 

probs=log.predict_proba(x_train)[:,1]

fpr, tpr, threshold=metrics.roc_curve(y_train,probs )
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='logistic')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()


# In[114]:


metrics.roc_auc_score(y_train,probs)


# In[115]:


model_report = pd.DataFrame()

tmp = pd.Series({'Model': " Logical Regression ",
                 'ROC Score' : metrics.roc_auc_score(y_test, newtest_pred),
                 'Precision Score': metrics.precision_score(y_test, newtest_pred),
                 'Recall Score': metrics.recall_score(y_test, newtest_pred),
                 'F1 Score' : metrics.f1_score(y_test, newtest_pred),
                 'Accuracy Score': metrics.accuracy_score(y_test, newtest_pred)})

model_logR_report = model_report.append(tmp, ignore_index = True)
model_logR_report


# In[ ]:




