
# coding: utf-8

# In[16]:


import sqlite3
import pandas as pd
conn = sqlite3.connect('factbook.db')
cursor = conn.cursor()

#Returns all tables and gives facts about them
q1 = "SELECT * FROM sqlite_master WHERE type='table';"
pd.read_sql_query(q1, conn)


# In[17]:


#Prints first 5 rows from "facts" table
q2 = 'SELECT * FROM facts LIMIT 5'
pd.read_sql_query(q2, conn)


# In[52]:


#Brief summary statistics on all of our countries
q3 = "SELECT MAX(birth_rate), MIN(birth_rate), MAX(death_rate), MIN(death_rate), MIN(population), MAX(population), MIN(population_growth), MAX(population_growth) FROM facts"
pd.read_sql_query(q3, conn)


# In[21]:


q4 = "SELECT * FROM facts WHERE (population = 0) OR (population > 7250000000)"
pd.read_sql_query(q4,conn)


# It appears that our database has Antarctica with a population of 0 and the entire world with a population of 7B+

# In[53]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

pop = "SELECT * FROM facts WHERE (population != 0) AND (population < 7250000000)"
df = pd.read_sql_query(pop,conn)

fig = plt.figure(figsize = (5,20))
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)

ax1.hist(df['population'], bins = 30, range = (0,150000000))
ax2.hist(df['population_growth'], bins=30, range = (0,5) )
ax3.hist(df['birth_rate'], bins=30, range = (0,50))
ax4.hist(df['death_rate'], bins=30, range = (0,15))
plt.show()

