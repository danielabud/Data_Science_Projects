
# coding: utf-8

# In[2]:


import pandas as pd
movies = pd.read_csv("fandango_score_comparison.csv")

movies


# In[3]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.hist(movies['Metacritic_norm_round'])
plt.title("Metacritic_norm_round")

plt.hist(movies['Fandango_Stars'])
plt.title("Fandango_Stars")


# It certainly appears that Fandango tends to score movies SIGNIFICANTLY higher than Metacritic, with most of their scores concentrated at 4.5 vs Metacritic's mode which is closer to 3. It appears that there is a fairly even distribution of scores on the Metacritic site, larger standard deviation, and

# In[4]:


import numpy as np

fand_mean = np.mean(movies['Fandango_Stars'])
fand_med = np.median(movies['Fandango_Stars'])
fand_std = np.std(movies['Fandango_Stars'])
meta_mean = np.mean(movies['Metacritic_norm_round'])
meta_med = np.median(movies['Metacritic_norm_round'])
meta_std = np.std(movies['Metacritic_norm_round'])
stats = [fand_mean,fand_med,fand_std,meta_mean,meta_med,meta_std]

print("fand_mean: " + str(fand_mean))
print("fand_med: " + str(fand_med))
print("fand_std: " + str(fand_std))
print("meta_mean: " + str(meta_mean))
print("meta_med: " + str(meta_med))
print("meta_std: " + str(meta_std))


# In[5]:


movies['fm_diff'] = movies['Fandango_Stars']-movies['Metacritic_norm_round']
movies['fm_diff'] = np.abs(movies['fm_diff'])

sorted_movies = movies.sort_values('fm_diff',ascending = False)

print(sorted_movies.head(5))


# In[10]:


import scipy.stats as stats

stats.pearsonr(movies['Fandango_Stars'],movies['Metacritic_norm_round'])

linreg = stats.linregress(movies['Fandango_Stars'],movies['Metacritic_norm_round'])

print(linreg)

def predict(metacritic_score):
    return metacritic_score*linreg[0]+linreg[1]

print(predict(3))
print(predict(1))
print(predict(5))

x =[1.0,5.0]
y = [predict(1),predict(5)]

plt.plot(x,y)
plt.show()

