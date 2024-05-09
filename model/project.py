import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data Load
df1=pd.read_csv("Bengaluru_House_Data.csv")
#print(df1.head())
#print(df1.shape)                               #Starting point the actual shape of the data frame is (13320,9)

                                                            #lets check the names of the columns
#print(df1.columns)                            #'area_type', 'availability', 'location', 'size', 'society' ,'total_sqft', 'bath', 'balcony', 'price' so this are the columns

#print(df1['area_type'].unique)             #Lets check the unique values in the column "area_type"
#'Super built-up  Area', 'Plot  Area', 'Built-up  Area', 'Carpet  Area'

#Lets see the no.of counts of unique values mention above
#print(df1['area_type'].value_counts())

#Drop the feature that are not required to build our model
df2=df1.drop(['area_type','society','balcony','availability'],axis='columns')

print(df2.shape)                            #Now the dimention of df2 i s(13320,5)

#Data Cleaning:Handle NA values

#Lets check the na values in each column
#print(df2.isnull().sum())

#Its not that big ammount of na values so drop it
df3=df2.dropna()
#lets again check the na counts in each column
#print(df3.isnull().sum()) #Now its zero in each column
print(df3.shape)                            #Now the dimention of df3 is (13246,5)




#Feature Engineering
#Adding new feature(integer) for bhk
df3['bhk']=df3['size'].apply(lambda x:int(x.split(' ')[0]))  #We just add a column that contain the first part of column number)

#print(df3.bhk.unique())  #The bhk contain 2,4,5,..... so on

#Explore total_sqft feature

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

#In this function we just try to checking the values which are other than a float (like this column also contain in ranges

#print(df3[~df3['total_sqft'].apply(is_float)].head(10)) #Printing the data which are other than float values

#Above shows that total_sqft can be a range (e.g. 2100-2850).
# For such case we can just take average of min and max value in the range.
# There are other cases such as 34.46Sq. Meter which one can convert to square ft using unit conversion.
# I am going to just drop such corner cases to keep things simple

def convert_sqft_to_num(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df4=df3.copy()   #Taking a copy of df3 in df4
df4.total_sqft=df4.total_sqft.apply(convert_sqft_to_num)
df4=df4[df4.total_sqft.notnull()]
#print(df4.head(2))
#print(df4.loc[30])



#Feature Engineering


#Add new feature called price per square feet
df5=df4.copy()
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
#print(df5.head())


df5_stats=df5['price_per_sqft'].describe()
#print(df5_stats)

df5.to_csv("bhp.csv",index=False)  #Tranferring the df5 to the new csv file named as bhp

#Examine locations which is a categorial variable.we need to apply dimentionality reduction technique here to reduce number of locations

df5.location=df5.location.apply(lambda x:x.strip())
location_stats=df5['location'].value_counts(ascending=False)#We are counting the no.of uniques locations
#print(location_stats)

#print(location_stats.values.sum())              #13200

#print(len(location_stats[location_stats>10]))    #240 #Checking the no.of location which have greater than the no.of 10
#print(len(location_stats))                                   #1287
#print(len(location_stats[location_stats<=10])) #1047



#Dimentionality Reduction

#We are going to label the location which are having less than
#10 data points should be tagged as "other" location.This way
#no.of categories can be reduced by huge ammount.
#Later on when we do one hot encoding,it will help us with having
#fewer dummy columns

location_stats_less_than_10=location_stats[location_stats<=10]
#print(location_stats_less_than_10)

#print(len(df5.location.unique()))
#labeling the location which are in the location_stats_less_than_10
df5.location=df5.location.apply(lambda x:'other' if x in location_stats_less_than_10 else x)
#print(len(df5.location.unique()))
#print(df5.head(10))



#Lets use business logic for outlier removal
#You ask business manager and he said that normally a square ft per bedroom is 300 i.e 2 bhk is minimum 600 sqft
#print(df5[df5.total_sqft/df5.bhk<300].head())
#printing the data points where total_sqft /bhk is less than 300


#Lets filter out above data point using "~" and then save it in the df6

df6=df5[~(df5.total_sqft/df5.bhk<300)]
print(df6.shape)                        #Now our df shape is (12456,7)




#Now lets remove outlier using mean and standard deviation
print(df6.price_per_sqft.describe())

#Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices.
#We should remove outliers per location using mean and one standard deviation

def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft)]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df7=remove_pps_outliers(df6)
print(df7.shape)

#We should check the prices of 2 and 3 BHK
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location)&(df.bhk==2)]
    bhk3=df[(df.location==location)&(df.bhk==3)]
    plt.figure(figsize=(15,10))   #Width 15 and height 10
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price(Lakh Indian Rupees)")
    plt.title(location)
    plt.legend(loc="upper right")
    plt.show()#Plotting a graph between the total square feet area and price

#plot_scatter_chart(df7,"Rajaji Nagar")  #We can clearly see that the some of the 2 bhk and 3 bhk points are overlapped this is clearly an error

#plot_scatter_chart(df7,"Hebbal") #same goes here

#Now we can remove those 2 BHK apartment whose price_per_sqft is less than mean
#price_per_sqft of 1 bhk appartment

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df8 = remove_bhk_outliers(df7)

#print(df8.shape)

#Plotting the same previous graph so that can we can recheck that our works well or not and fix this issue
#Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties

#plot_scatter_chart(df8,"Rajaji Nagar")
#plot_scatter_chart(df8,"Hebbal")

#Based on above charts we can see that data points highlighted in red below are outliers and
#they are being removed due to remove_bhk_outliers function

import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
#plt.show()

#Outlier Removal using Bathroom Feature

#print(df8.bath.unique())
plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("No. of Bathrooms")
plt.ylabel("Count")
#plt.show()

#print(df8[df8.bath>10])                 #It is unusal to have 2 more bedrooms than number of bedroom in a room

#print(df8[df8.bath>df8.bhk+2])

#total bath=total bed+1 max

df9=df8[df8.bath<df8.bhk+2]
#print(df9.shape)                #(7239,7)

#print(df9.head(2))

df10=df9.drop(['size','price_per_sqft'],axis='columns')
#print(df10.head(3))

#Use One Hot Encoding For Location
dummies=pd.get_dummies(df10.location)
#print(dummies.head(3))
#we concat the df10 and dummies (after dropping the
df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
#print(df11.head())

df12=df11.drop('location',axis='columns')
#print(df12.head(2))
#Dropping the location then save it in df12

#Build A model now

#print(df12.shape)

x=df12.drop(['price'],axis='columns')
#print(x.head(3))            #x contain all the all the contain columns of df12 except the price


y=df12.price                #y contains price column
#print(y.head(3))

#splitting the data into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)


#Training or dataset using linear regression
from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(x_train,y_train)
print(lr_clf.score(x_test,y_test))

#Use K fold cross validation to measure accuracy of our Linear Regression Model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
print(cross_val_score(LinearRegression(),x,y,cv=cv))

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

def find_best_model_using_gridsearchcv(x, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],
                'copy_X': [True, False],
                'n_jobs': [None, 1, 2, 4]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['poisson', 'squared_error', 'absolute_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Assuming x and y are defined before calling the function
print(find_best_model_using_gridsearchcv(x, y))

#Testing the model for few properties


def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(x.columns==location)[0][0]

    z = np.zeros(len(x.columns))
    z[0] = sqft
    z[1] = bath
    z[2] = bhk
    if loc_index >= 0:
        z[loc_index] = 1

    return lr_clf.predict([z])[0]

print(predict_price('1st Phase JP Nagar',1000,2,2))
print(predict_price('1st Phase JP Nagar',1000,3,3))
print(predict_price('Indira Nagar',1000,2,2))
print(predict_price('Indira Nagar',1000,3,3))


import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


