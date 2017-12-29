
# coding: utf-8

# In[32]:


import pandas as pd
star_wars = pd.read_csv("star_wars.csv", encoding="ISO-8859-1")

#star_wars.columns
#Need to delete rows with no RespondentID value
star_wars = star_wars[pd.notnull(star_wars['RespondentID'])]
star_wars.head(10)


# In[33]:


#To find out what all the answer choices were for the next two questions,
#we can use value_counts. We note that all are Yes/No/NaN
star_wars['Have you seen any of the 6 films in the Star Wars franchise?'].value_counts()
star_wars['Do you consider yourself to be a fan of the Star Wars film franchise?'].value_counts()

#let's use a map function to convert Yes to True and No to False
yes_no = {
    "Yes": True,
    "No": False
}
star_wars['Have you seen any of the 6 films in the Star Wars franchise?'] = star_wars['Have you seen any of the 6 films in the Star Wars franchise?'].map(yes_no)
star_wars['Do you consider yourself to be a fan of the Star Wars film franchise?'] = star_wars['Do you consider yourself to be a fan of the Star Wars film franchise?'].map(yes_no)


# In[34]:


#this cell changes the column answers from a string to a True/False
#value depending on the answer

import numpy as np
ep_bool = {
    "Star Wars: Episode I  The Phantom Menace": True,
    "Star Wars: Episode II  Attack of the Clones": True,
    "Star Wars: Episode III  Revenge of the Sith": True,
    "Star Wars: Episode IV  A New Hope": True,
    "Star Wars: Episode V The Empire Strikes Back": True,
    "Star Wars: Episode VI Return of the Jedi": True,
    np.nan: False
}


for episode in star_wars.columns[3:9]:
    star_wars[episode] = star_wars[episode].map(ep_bool)


# In[35]:


#this cell renames the columns to something more manageable

star_wars = star_wars.rename(columns = {
    "Which of the following Star Wars films have you seen? Please select all that apply.": "seen ep1",
    "Unnamed: 4": "seen ep2",
    "Unnamed: 5": "seen ep3",
    "Unnamed: 6": "seen ep4",
    "Unnamed: 7": "seen ep5",
    "Unnamed: 8": "seen ep6"
})


# In[36]:


#convert "most favorite" and "least favorite" movie preferences to numeric
star_wars[star_wars.columns[9:15]] = star_wars[star_wars.columns[9:15]].astype(float)

star_wars = star_wars.rename(columns = {
    "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film." : "ep1 rank",
    "Unnamed: 10": "ep2 rank",
    "Unnamed: 11": "ep3 rank",
    "Unnamed: 12": "ep4 rank",
    "Unnamed: 13": "ep5 rank",
    "Unnamed: 14": "ep6 rank"
})

#compute the mean of each column
col_avg = star_wars.mean()
print(col_avg)


# In[37]:


import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
#lets plot col_avg[6:12]

plt.bar(range(6), star_wars[star_wars.columns[9:15]].mean())
plt.title("Worst Star Wars Episode - higher is worse")


# In[38]:


plt.title("Who watched what episode?")
plt.bar(range(6), star_wars[star_wars.columns[3:9]].sum())


# In[40]:


#Now let's take a look at some trends

males = star_wars[star_wars["Gender"] == "Male"]
females = star_wars[star_wars["Gender"] == "Female"]


# In[44]:


plt.bar(range(6), females[females.columns[9:15]].mean())
plt.title("FEMALE ranking of worst Star Wars movies (higher is worse)")
plt.show()


print(males[males.columns[9:15]].columns)
plt.bar(range(6), males[males.columns[9:15]].mean())
plt.title("MALE ranking of worst Star Wars movies (higher is worse)")
plt.show()

plt.bar(range(6), females[females.columns[3:9]].sum())
plt.title("FEMALE ranking of most viewed episodes of Star Wars")
plt.show()

plt.bar(range(6), males[males.columns[3:9]].sum())
plt.title("MALE ranking of most viewed episodes of Star Wars")
plt.show()


