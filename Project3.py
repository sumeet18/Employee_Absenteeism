
# coding: utf-8

# In[1]:


#Load libraries
import os
import pandas as pd
import numpy as np


# In[2]:


#Set working directory
os.chdir("G:\Edwisor")


# In[3]:


#load data from excel
data = pd.read_excel("G:\Edwisor\data.xls", sep = ',', encoding = 'Latin-1')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


#see no of rows and columns
data.shape


# In[8]:


#number of unique values in all variables
data.nunique()


# In[9]:


data = data.rename(columns = {'Reason for absence':'reason','Month of absence':'month','Day of the week':'day',
                              'Transportation expense':'expense', 'Distance from Residence to Work':'distance',
                              'Service time':'time','Work load Average/day':'workload','Hit target':'target',
                              'Disciplinary failure':'discipline','Social drinker':'drinker','Social smoker':'smoker',
                              'Body mass index':'bmi','Absenteeism time in hours':'absentime'})


# In[10]:


data['reason'] = data['reason'].astype(object)
data['month'] = data['month'].astype(object)
data['day'] = data['day'].astype(object)
data['discipline'] = data['discipline'].astype(object)
data['drinker'] = data['drinker'].astype(object)
data['smoker'] = data['smoker'].astype(object)
data['ID'] = data['ID'].astype(object)
data['Pet'] = data['Pet'].astype(object)
data['Son'] = data['Son'].astype(object)
data['Seasons'] = data['Seasons'].astype(object)
data['Education'] = data['Education'].astype(object)


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


#bargraph
var = data.groupby('ID').ID.count() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('IDs')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[13]:


#bargraph
var = data.groupby('reason').reason.count() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('IDs')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[14]:


#bargraph
var = data.groupby('month').month.count() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('IDs')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[15]:


#bargraph
var = data.groupby('Seasons').Seasons.count() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('IDs')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[16]:


#bargraph
var = data.groupby('day').day.count() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('IDs')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[17]:


#bargraph
var = data.groupby('Pet').Pet.count() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('No.ofPets')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[18]:


#bargraph
var = data.groupby('drinker').drinker.count() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('No. of Social drinker')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[19]:


#bargraph
var = data.groupby('smoker').smoker.count() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('No. of Social smoker')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[20]:


#bargraph
var = data.groupby('Education').Pet.count() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('No. of Education')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[21]:


#bargraph
var = data.groupby('discipline').discipline.count() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('No. of Disciplinary failure ')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[22]:


#bargraph
var = data.groupby('Son').Son.count() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('No. of Disciplinary failure ')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[23]:


#Create dataframe with missing percentage
missing_val = pd.DataFrame(data.isnull().sum())

#Reset index
missing_val = missing_val.reset_index()

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(data))*100

#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)


# In[24]:


missing_val


# In[25]:


# Droping observation in which "Absenteeism time in hours" has missing value
data = data.drop(data[data['absentime'].isnull()].index, axis=0)
print(data.shape)
print(data['absentime'].isnull().sum())


# In[26]:


data['bmi'].iloc[70]


# In[27]:


# Checking for "Body mass index" column
# Actual value = 27

#create missing value
data['bmi'].iloc[70] = np.nan


# In[28]:


data['bmi'].iloc[70]


# In[29]:


# # Impute with mean
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())
data['bmi'].iloc[70]


# In[30]:


# Checking for "Body mass index" column
# Actual value = 27
# Mean = 26.706

#create missing value
data['bmi'].iloc[70] = np.nan


# In[31]:


data['bmi'].iloc[70]


# In[32]:


# # Impute with median
data['bmi'] = data['bmi'].fillna(data['bmi'].median())
data['bmi'].iloc[70]


# In[33]:


# Checking for "Body mass index" column
# Actual value = 27
# Mean = 26.706
# Median = 25

#create missing value
data['bmi'].iloc[70] = np.nan


# In[34]:


data['bmi'].iloc[70]


# In[ ]:


#Apply KNN imputation algorithm

data = pd.DataFrame(KNN(k = 3).complete(data), columns = data.columns)


# In[180]:


#Load csv data in python
new = pd.read_csv("new.csv", sep = ',')


# In[181]:


new.info()


# In[182]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Transportation.expense'])
plt.xlabel('Transportation.expense')


# In[183]:


#calculate 99th %ile
a = np.array(new['Transportation.expense'])
p = np.percentile(a, 99) # return 99th percentile,
print(p)


# In[184]:


#replace values in the variable
for i in range(len(new)):
      if  new['Transportation.expense'].loc[i]>378:
          new['Transportation.expense'].loc[i]=378


# In[185]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Transportation.expense'])
plt.xlabel('Transportation.expense')


# In[186]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Distance.from.Residence.to.Work'])
plt.xlabel('Distance.from.Residence.to.Work')


# In[187]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Service.time'])
plt.xlabel('Service.time')


# In[188]:


#calculate 99th %ile
a = np.array(new['Service.time'])
p = np.percentile(a, 99) # return 99th percentile,
print(p)


# In[189]:


#replace values in the variable
for i in range(len(new)):
      if  new['Service.time'].loc[i]>18:
          new['Service.time'].loc[i]=18


# In[190]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Service.time'])
plt.xlabel('Service.time')


# In[191]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Age'])
plt.xlabel('Age')


# In[192]:


#calculate 99th %ile
a = np.array(new['Age'])
p = np.percentile(a, 98) # return 99th percentile,
print(p)


# In[193]:


#replace values in the variable
for i in range(len(new)):
      if  new['Age'].loc[i]>50:
          new['Age'].loc[i]=50


# In[194]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Age'])
plt.xlabel('Age')


# In[195]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Work.load.Average.day.'])
plt.xlabel('Work.load.Average.day. ')


# In[196]:


#calculate 99th %ile
a = np.array(new['Work.load.Average.day.'])
p = np.percentile(a, 94) # return 99th percentile,
print(p)


# In[197]:


#replace values in the variable
for i in range(len(new)):
      if  new['Work.load.Average.day.'].loc[i]>343253:
          new['Work.load.Average.day.'].loc[i]=343253


# In[198]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Work.load.Average.day.'])
plt.xlabel('Work.load.Average.day. ')


# In[199]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Hit.target'])
plt.xlabel('Hit.target')


# In[200]:


#calculate 1st %ile
a = np.array(new['Hit.target'])
p = np.percentile(a, 4) # return 1st percentile,
print(p)


# In[201]:


#replace values in the variable
for i in range(len(new)):
      if  new['Hit.target'].loc[i]<87:
          new['Hit.target'].loc[i]=87


# In[202]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Hit.target'])
plt.xlabel('Hit.target')


# In[203]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Weight'])
plt.xlabel('Weight')


# In[204]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Height'])
plt.xlabel('Height')


# In[205]:


#calculating 1st and 99th %ile
a = np.array(new['Height'])
q = np.percentile(a, 1) 
p = np.percentile(a, 84) 
print(q)
print(p)


# In[206]:


#replace values in the variable
for i in range(len(new)):
      if  new['Height'].loc[i]<165:
          new['Height'].loc[i]=165
      if  new['Height'].loc[i]>175:
          new['Height'].loc[i]=175


# In[207]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Height'])
plt.xlabel('Height')


# In[208]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(new['Body.mass.index'])
plt.xlabel('Body.mass.index')


# In[65]:


cnames= ['Transportation.expense','Hit.target','Distance.from.Residence.to.Work','Service.time','Age',
'Work.load.Average.day.','Weight','Height','Body.mass.index','Absenteeism.time.in.hours']


# In[66]:


import seaborn as sns


# In[67]:


#correlation
df_corr = new.loc[:,cnames]
#correlation plot
f, ax = plt.subplots(figsize=(12,8))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[68]:


categorical = ['ID','Reason.for.absence','Month.of.absence','Day.of.the.week',
                     'Seasons','Disciplinary.failure', 'Education', 'Social.drinker',
                     'Social.smoker', 'Pet', 'Son']


# In[69]:


from scipy import stats


# In[70]:


#loop for ANOVA test Since the target variable is continuous
for i in categorical:
    f, p = stats.f_oneway(new[i], new["Absenteeism.time.in.hours"])
    print("P value for variable "+str(i)+" is "+str(p))


# In[71]:


#histogram
plt.hist(new['Transportation.expense'], bins='auto')
plt.xlabel('Transportation.expense')
plt.ylabel("Frequency")


# In[72]:


#histogram
plt.hist(new['Hit.target'], bins='auto')
plt.xlabel('Hit.target')
plt.ylabel("Frequency")


# In[73]:


#histogram
plt.hist(new['Distance.from.Residence.to.Work'], bins='auto')
plt.xlabel('Distance.from.Residence.to.Work')
plt.ylabel("Frequency")


# In[74]:


#histogram
plt.hist(new['Service.time'], bins='auto')
plt.xlabel('Service.time')
plt.ylabel("Frequency")


# In[75]:


#histogram
plt.hist(new['Age'], bins='auto')
plt.xlabel('Age')
plt.ylabel("Frequency")


# In[76]:


#histogram
plt.hist(new['Work.load.Average.day.'], bins='auto')
plt.xlabel('Work.load.Average.day.')
plt.ylabel("Frequency")


# In[77]:


#histogram
plt.hist(new['Weight'], bins='auto')
plt.xlabel('Weight')
plt.ylabel("Frequency")


# In[78]:


#histogram
plt.hist(new['Height'], bins='auto')
plt.xlabel('Height')
plt.ylabel("Frequency")


# In[79]:


#histogram
plt.hist(new['Body.mass.index'], bins='auto')
plt.xlabel('Body.mass.index')
plt.ylabel("Frequency")


# In[209]:


cnames= ['Transportation.expense','Hit.target','Distance.from.Residence.to.Work','Service.time','Age',
'Work.load.Average.day.','Weight','Height','Body.mass.index']


# In[210]:


# #Normalization
for i in cnames:
    if i == 'Absenteeism.time.in.hours':
        continue
    new[i] = (new[i] - new[i].min())/(new[i].max()-new[i].min())


# In[211]:


new.head()


# In[212]:


# Get dummy variables for categorical variables
new = pd.get_dummies(data = new, columns = categorical)


# In[213]:


new.head()


# In[214]:


#delete variable
del new['ID_36']
del new['Reason.for.absence_0']
del new['Month.of.absence_0']
del new['Day.of.the.week_2']
del new['Seasons_4']
del new['Education_4']
del new['Disciplinary.failure_1']
del new['Son_4']
del new['Social.drinker_0']
del new['Social.smoker_0']
del new['Pet_8']


# In[215]:


columnsTitles = ['Transportation.expense', 'Distance.from.Residence.to.Work',
       'Service.time', 'Age', 'Work.load.Average.day.', 'Hit.target', 'Weight',
       'Height', 'Body.mass.index','Social.drinker_1','ID_1','ID_2','ID_3','ID_4','ID_5','ID_6','ID_7','ID_8','ID_9','ID_10',
                 'ID_11','ID_12','ID_13','ID_14','ID_15','ID_16','ID_17','ID_18','ID_19','ID_20','ID_21','ID_22','ID_23',
                 'ID_24','ID_25','ID_26','ID_27''ID_28','ID_30','ID_29','ID_31','ID_32','ID_33','ID_34','ID_35',
                 'Reason.for.absence_1','Reason.for.absence_2','Reason.for.absence_3','Reason.for.absence_4',
                 'Reason.for.absence_5','Reason.for.absence_6','Reason.for.absence_7','Reason.for.absence_8',
                 'Reason.for.absence_9','Reason.for.absence_10','Reason.for.absence_11','Reason.for.absence_12',
                 'Reason.for.absence_13','Reason.for.absence_14','Reason.for.absence_15','Reason.for.absence_16',
                 'Reason.for.absence_17','Reason.for.absence_18','Reason.for.absence_19','Reason.for.absence_20',
                 'Reason.for.absence_21','Reason.for.absence_22','Reason.for.absence_23','Reason.for.absence_24',
                 'Reason.for.absence_25','Reason.for.absence_26','Reason.for.absence_27','Reason.for.absence_28',
                 'Month.of.absence_1','Month.of.absence_2','Month.of.absence_3','Month.of.absence_4','Month.of.absence_5',
                 'Month.of.absence_6','Month.of.absence_7','Month.of.absence_8','Month.of.absence_9','Month.of.absence_10',
                 'Month.of.absence_11','Month.of.absence_12','Day.of.the.week_3','Day.of.the.week_4','Day.of.the.week_5',
                 'Day.of.the.week_6','Education_1','Education_2','Education_3','Disciplinary.failure_0',
                 'Seasons_1','Seasons_2','Seasons_3','Social.smoker_1', 'Pet_0', 'Pet_1', 'Pet_2', 'Pet_4', 'Pet_5', 'Son_0',
       'Son_1', 'Son_2', 'Son_3','Absenteeism.time.in.hours']


# In[216]:


new = new.reindex(columns=columnsTitles)


# In[217]:


new.info()


# In[90]:


#bargraph
var = data.groupby('ID').absentime.mean() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('No. of ID ')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[91]:


#bargraph
var = data.groupby('reason').absentime.mean() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('No. of ID ')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[92]:


#bargraph
var = data.groupby('Seasons').absentime.mean() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('No. of ID ')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[94]:


#bargraph
var = data.groupby('month').absentime.mean() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('No. of ID ')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[95]:


#bargraph
var = data.groupby('day').absentime.mean() 
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('No. of ID ')
ax1.set_ylabel('count')
ax1.set_title("frequency")
var.plot(kind='bar')


# In[96]:


#Import Libraries for decision tree
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[220]:


#divide train and test data
train, test = train_test_split(new, test_size=0.2)


# In[221]:


#desicion tree regression
fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:105], train.iloc[:,105])


# In[107]:


#apply on test data
predictions_DT = fit_DT.predict(test.iloc[:,0:105])


# In[108]:


predictions_DT


# In[109]:


test.iloc[:,0:104]


# In[110]:


test.iloc[:,105]


# In[111]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(test.iloc[:,105], predictions_DT))  
print('Mean Squared Error:', metrics.mean_squared_error(test.iloc[:,105], predictions_DT))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test.iloc[:,105], predictions_DT)))


# In[112]:


#Calculate MAPE
def MAPE(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred) / y_true))*100
    return mape


# In[113]:


MAPE(test.iloc[:,105], predictions_DT)


# In[114]:


# ## Linear Regression
#Import libraries for LR
import statsmodels.api as sm


# In[115]:


y = train.iloc[:,105]
x = train.iloc[:,0:105]


# In[116]:


# Train the model using the training sets
model = sm.OLS(y, x.astype(float)).fit()


# In[117]:


# Print out the statistics
model.summary()


# In[118]:


# make the predictions by the model
predictions_LR = model.predict(test.iloc[:,0:105]) 


# In[119]:


MAPE(test.iloc[:,105], predictions_LR)


# In[120]:


print('Mean Absolute Error:', metrics.mean_absolute_error(test.iloc[:,105], predictions_LR))  
print('Mean Squared Error:', metrics.mean_squared_error(test.iloc[:,105], predictions_LR))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test.iloc[:,105], predictions_LR)))


# In[121]:


#randomforest model

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(train.iloc[:,0:105], train.iloc[:,105])  
y_pred = regressor.predict(test.iloc[:,0:105]) 


# In[122]:


print('Mean Absolute Error:', metrics.mean_absolute_error(test.iloc[:,105], y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(test.iloc[:,105], y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test.iloc[:,105], y_pred)))


# In[123]:


MAPE(test.iloc[:,105], y_pred)


# In[132]:


new.info()


# In[218]:


missing_val = pd.DataFrame(new.isnull().sum())


# In[219]:


missing_val

