
# coding: utf-8

# We are going to create a simple k-nearest model to predict car prices.
#     First, lets import data and see what it looks like.

# In[2]:


import pandas as pd

head = ['symboling','normalized_losses','make','fuel_type','aspiration','num_doors','body_style','drive_wheels','engine_location','wheel_base','length','width','height','curb_weight','engine_type','num_cylinders','engine_size','fuel_system','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price']
cars = pd.read_csv("imports-85.data", names = head)
cars


# We're going to eliminate all "?" from our data and replace with null values before converting all numerical values into either floats or int. We can only use numerical values to model. 

# In[3]:


####################################################################################################

#I would like to effectively use a dictionary to add an extra layer to our analysis by converting strings to numerical functions

#subs = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'twelve':12}

#for n, doors in enumerate(cars['num_doors']):
 #   cars['num_doors'][n] = 
    

#numerical_strings = ['num_doors','num_cylinders']


# In[4]:


import numpy as np

cars = cars.replace('?', np.nan)

numeric = ['normalized_losses','wheel_base','length','width','height','curb_weight','engine_size','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price']
non_numeric = ['symboling','make','fuel_type','aspiration','num_doors','body_style','drive_wheels','engine_location','engine_type','num_cylinders','fuel_system']

numeric_cars = cars[numeric]

numeric_cars = numeric_cars.astype('float64')

percent_missing_normalized_losses = cars['normalized_losses'].isnull().sum()/len(cars['normalized_losses'])
print(percent_missing_normalized_losses)


# Notice that we are missing 20% of normalized_losses. I believe the best way to handle this data would be to simply  drop these individual values, but for simplicity we will drop the entire row for those that have missing values. We then need to normalize each set of values to numbers between 0 and 1 (except price).

# In[5]:


numeric_cars = numeric_cars.dropna(axis = 0, how = 'any')
price = numeric_cars['price']

#let's normalize all values to range from 0 to 1
numeric_cars = (numeric_cars - numeric_cars.min())/(numeric_cars.max() - numeric_cars.min())
numeric_cars['price'] = price
print(numeric_cars.isnull().sum())
numeric_cars


# In[6]:


def knn_train_test(train_name,target_name,dataframe):
    #shuffle the index only
    shuffled_index = np.random.permutation(dataframe.index)
    #reindex from the shuffled index - now we have a random DF!
    rand_df = dataframe.reindex(shuffled_index)    
    length = len(dataframe)
    train_length = int(length/2)
    #iloc is used to ensure we are selecting row indeces
    train_set = rand_df.iloc[:train_length]
    test_set = rand_df.iloc[train_length:]    
    
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    knn = KNeighborsRegressor()

    #instantiate machine learning model
    knn = KNeighborsRegressor(n_neighbors=5,algorithm = 'brute')
    #create model
    knn.fit(train_set[[train_name]],train_set[target_name])
    #use model to predict prices
    prediction = knn.predict(test_set[[train_name]])
    #compare prices to actual prices
    mse = mean_squared_error(test_set[target_name], prediction)
    rmse = mse**.5
    return(rmse)
 

#Now lets get an average of different RMSE values
#rmse_vals = []
#for i in range(100):
#    rmse = knn_train_test('horsepower','price',numeric_cars)
#    rmse_vals.append(rmse)
#print(np.mean(rmse_vals))
#now that we have that working, let's try it with different columns!!
numeric = ['normalized_losses','wheel_base','length','width','height','curb_weight','engine_size','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price']

column_rmse = {}
for column in numeric:
    for i in range(20):
        rmse = knn_train_test(column,'price',numeric_cars)
    column_rmse[column] = rmse
print(column_rmse)

max(v for k, v in column_rmse.items() if v != 0)


# Using just the above code, we are unable to determine which column is best because there is some variability each time we run our code. To avoid this, we need to get an "average" of multiple runs by adding in the ability to choose the number of times to run our test.

# In[23]:


def knn_train_test(train_name,target_name,dataframe,k):
    #shuffle the index only
    shuffled_index = np.random.permutation(dataframe.index)
    #reindex from the shuffled index - now we have a random DF!
    rand_df = dataframe.reindex(shuffled_index)    
    length = len(dataframe)
    train_length = int(length/2)
    #iloc is used to ensure we are selecting row indeces
    train_set = rand_df.iloc[:train_length]
    test_set = rand_df.iloc[train_length:]    
    
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    knn = KNeighborsRegressor()
    
    set_of_rmse = []
    for i in range(k):
        #instantiate machine learning model
        knn = KNeighborsRegressor(n_neighbors=5,algorithm = 'brute')
        #create model
        knn.fit(train_set[[train_name]],train_set[target_name])
        #use model to predict prices
        prediction = knn.predict(test_set[[train_name]])
        #compare prices to actual prices
        mse = mean_squared_error(test_set[target_name], prediction)
        set_of_rmse.append(mse**.5)
    return(np.mean(set_of_rmse))
 
numeric = ['normalized_losses','wheel_base','length','width','height','curb_weight','engine_size','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price']

my_data = pd.DataFrame(index=[1,3,5,7,9])
for column in numeric:
    my_data[column]=np.nan
    for i in range(0,5):
        rmse = knn_train_test(column,'price',numeric_cars,2*i+1)
        my_data[column][2*i+1]=rmse
print(my_data)

#Now let's visualize our data


# In[8]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

i=0
x_vals = [1,3,5,7,9]
for column in my_data.columns:
    i=i+1
    fig = plt.figure(figsize =(20,40))
    ax = fig.add_subplot(8,2,i)
    ax.set_xlim(0,10)
    plt.grid(True)
    ax.scatter(x_vals,my_data[column])
    plt.show()
    
for column in my_data.columns:
    avg = np.mean(my_data[column])
    print(column,avg)


# Now that we've been able to plot different RMSEs based on the number of times we iterate, let's make our model learn from ALL The data as opposed to individual columns.

# In[9]:


def knn_train_test(train_names,target_name,dataframe):
    shuffled_index = np.random.permutation(dataframe.index)
    rand_df = dataframe.reindex(shuffled_index)    
    length = len(dataframe)
    train_length = int(length/2)
    train_set = rand_df.iloc[:train_length]
    test_set = rand_df.iloc[train_length:]    
    
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    knn = KNeighborsRegressor()
    knn = KNeighborsRegressor(n_neighbors=5,algorithm = 'brute')
    #NOTE: here we are modifying the data by removing a [] so that it can take a list of names rather than a string
    knn.fit(train_set[train_names],train_set[target_name])
    prediction = knn.predict(test_set[train_names])
    mse = mean_squared_error(test_set[target_name], prediction)
    rmse = mse**.5
    return(rmse)
 
numeric = ['normalized_losses','wheel_base','length','width','height','curb_weight','engine_size','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price']

rmse = knn_train_test(numeric,'price',numeric_cars)
print(rmse)


# In order from "best" to "worst" features:
# width
# curb_weight
# engine_size
# highway_mpg
# horsepower

# In[10]:


two_rmse = knn_train_test(['width','curb_weight'],'price',numeric_cars)
three_rmse = knn_train_test(['width','curb_weight','engine_size'],'price',numeric_cars)
four_rmse = knn_train_test(['width','curb_weight','engine_size','highway_mpg'],'price',numeric_cars)
five_rmse = knn_train_test(['width','curb_weight','engine_size','highway_mpg','horsepower'],'price',numeric_cars)

print(two_rmse,three_rmse,four_rmse,five_rmse)


# We see from the above information that using only the best THREE or FOUR variables yields the best results - above 3-4, we actually get a higher error/worse model. This is expected. Now lets modify how many neighbors we are looking at to see if we can get good results.

# In[30]:


def knn_train_test(train_names,target_name,dataframe,k):
    #shuffle the index only
    shuffled_index = np.random.permutation(dataframe.index)
    #reindex from the shuffled index - now we have a random DF!
    rand_df = dataframe.reindex(shuffled_index)    
    length = len(dataframe)
    train_length = int(length/2)
    #iloc is used to ensure we are selecting row indeces
    train_set = rand_df.iloc[:train_length]
    test_set = rand_df.iloc[train_length:]    
    
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    knn = KNeighborsRegressor()
    
    set_of_rmse = []
    for i in range(k):
        #instantiate machine learning model
        knn = KNeighborsRegressor(n_neighbors=k,algorithm = 'brute')
        #create model
        knn.fit(train_set[train_names],train_set[target_name])
        #use model to predict prices
        prediction = knn.predict(test_set[train_names])
        #compare prices to actual prices
        mse = mean_squared_error(test_set[target_name], prediction)
        set_of_rmse.append(mse**.5)
    return(np.mean(set_of_rmse))
 
numeric = ['normalized_losses','wheel_base','length','width','height','curb_weight','engine_size','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price']

best_fit_cols1 = ['width','curb_weight','engine_size','highway_mpg']
best_fit_cols2 = ['width','curb_weight','engine_size','highway_mpg','horsepower']
best_fit_cols3 = ['width','curb_weight','engine_size']

rmses1 = []
rmses2 = []
rmses3 = []
for i in range(25):
    temp_rmse1 = knn_train_test(best_fit_cols1,'price',numeric_cars,i+1)
    rmses1.append(temp_rmse1)
    temp_rmse2 = knn_train_test(best_fit_cols2,'price',numeric_cars,i+1)
    rmses2.append(temp_rmse2)
    temp_rmse3 = knn_train_test(best_fit_cols3,'price',numeric_cars,i+1)
    rmses3.append(temp_rmse3)
    
get_ipython().magic('matplotlib inline')

x_vals = range(1,26)

fig = plt.figure(figsize =(8,4))
plt.plot(x_vals,rmses1)
plt.plot(x_vals,rmses2)
plt.plot(x_vals,rmses3)

