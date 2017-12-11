
# coding: utf-8

# # Read in the data

# In[2]:


import pandas
import numpy
import re

data_files = ["ap_2010.csv", "class_size.csv", "demographics.csv",
              "graduation.csv", "hs_directory.csv", "sat_results.csv"]
data = {}
#reads CSV and creates a spot in, for example data[ap_2010]
for f in data_files:
    d = pandas.read_csv("schools/{0}".format(f))
    data[f.replace(".csv", "")] = d


# # Read in the surveys

# In[3]:


all_survey = pandas.read_csv("schools/survey_all.txt", delimiter="\t", encoding='windows-1252')
d75_survey = pandas.read_csv("schools/survey_d75.txt", delimiter="\t", encoding='windows-1252')
#combines data sets into survey so there's only 1
survey = pandas.concat([all_survey, d75_survey], axis=0)

survey["DBN"] = survey["dbn"]

#lists the columns of interest in survey
survey_fields = [
    "DBN", 
    "rr_s", 
    "rr_t", 
    "rr_p", 
    "N_s", 
    "N_t", 
    "N_p", 
    "saf_p_11", 
    "com_p_11", 
    "eng_p_11", 
    "aca_p_11", 
    "saf_t_11", 
    "com_t_11", 
    "eng_t_10", 
    "aca_t_11", 
    "saf_s_11", 
    "com_s_11", 
    "eng_s_11", 
    "aca_s_11", 
    "saf_tot_11", 
    "com_tot_11", 
    "eng_tot_11", 
    "aca_tot_11",
]

#renames survey to only be fields of interest from above,
#inserts it into the data DF
survey = survey.loc[:,survey_fields]
data["survey"] = survey


# # Add DBN columns

# In[4]:


data["hs_directory"]["DBN"] = data["hs_directory"]["dbn"]

#DBN column needs to be 2 values wide, but sometimes is listed
#as only 1. Here we insert a 0 if it's only 1 value wide
def pad_csd(num):
    string_representation = str(num)
    if len(string_representation) > 1:
        return string_representation
    else:
        return "0" + string_representation
    
data["class_size"]["padded_csd"] = data["class_size"]["CSD"].apply(pad_csd)
data["class_size"]["DBN"] = data["class_size"]["padded_csd"] + data["class_size"]["SCHOOL CODE"]


# # Convert columns to numeric

# In[5]:


cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
#converts SAT score subsets to numeric so we can find
#a cumulative SAT score
for c in cols:
    data["sat_results"][c] = pandas.to_numeric(data["sat_results"][c], errors="coerce")

#adds all sat scores together into sat_score
data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] + data['sat_results'][cols[2]]

#from RE (regular expressions) library, extracts either lat or lon
#by finding open parenthesis, then any values, then a comma, then 
#any other values, then a close parenthesis. It then splits that
#coordinate such as (43.123,-103.761) into two strings and gets rid
#of the parenthesis before returning the first or 2nd value
def find_lat(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lat = coords[0].split(",")[0].replace("(", "")
    return lat

def find_lon(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lon = coords[0].split(",")[1].replace(")", "").strip()
    return lon


data["hs_directory"]["lat"] = data["hs_directory"]["Location 1"].apply(find_lat)
data["hs_directory"]["lon"] = data["hs_directory"]["Location 1"].apply(find_lon)

data["hs_directory"]["lat"] = pandas.to_numeric(data["hs_directory"]["lat"], errors="coerce")
data["hs_directory"]["lon"] = pandas.to_numeric(data["hs_directory"]["lon"], errors="coerce")


# # Condense datasets

# In[6]:


#only chooses data sets from grades 9-12 and program type "Gen Ed"
class_size = data["class_size"]
class_size = class_size[class_size["GRADE "] == "09-12"]
class_size = class_size[class_size["PROGRAM TYPE"] == "GEN ED"]

#groups class size by DBN and averages the class sizes for that DBN
class_size = class_size.groupby("DBN").agg(numpy.mean)
#resets index. it used to be DBN but now will be 0, 1, 2, etc.
class_size.reset_index(inplace=True)
data["class_size"] = class_size

#filters demographic data by school year (only 2011-2012)
data["demographics"] = data["demographics"][data["demographics"]["schoolyear"] == 20112012]
#more filtering
data["graduation"] = data["graduation"][data["graduation"]["Cohort"] == "2006"]
data["graduation"] = data["graduation"][data["graduation"]["Demographic"] == "Total Cohort"]


# # Convert AP scores to numeric

# In[7]:


cols = ['AP Test Takers ', 'Total Exams Taken', 'Number of Exams with scores 3 4 or 5']
#converts AP scores into numeric data, ignoring rows with NaN (blanks)
for col in cols:
    data["ap_2010"][col] = pandas.to_numeric(data["ap_2010"][col], errors="coerce")


# # Combine the datasets

# In[8]:


combined = data["sat_results"]

#merges ap and sat_results data frames based on DBN. Will use
#DBN only from 1st DF (sat_results) since "how" is set to the first
#data frame, or "left" data frame
combined = combined.merge(data["ap_2010"], on="DBN", how="left")
combined = combined.merge(data["graduation"], on="DBN", how="left")

to_merge = ["class_size", "demographics", "survey", "hs_directory"]
#now since we have higher quality data, we will only look at data
#that has both DBNs from combined AND data[m]
for m in to_merge:
    combined = combined.merge(data[m], on="DBN", how="inner")

combined = combined.fillna(combined.mean())
combined = combined.fillna(0)


# # Add a school district column for mapping

# In[9]:


#sometimes these dictionaries have more than 2 values for DBN but
#we only want first two values:
def get_first_two_chars(dbn):
    return dbn[0:2]

combined["school_dist"] = combined["DBN"].apply(get_first_two_chars)


# # Find correlations

# In[10]:


#although correlations don't tell us everything, they are a good place
#to start! combined.corr() gets correlation from each column in a
#matrix sstyle format
correlations = combined.corr()

correlations = correlations["sat_score"]
print(correlations)


# # Plotting values of interest

# In[11]:


import matplotlib.pyplot as plt
# renders the figure in a notebook instead of in dump
get_ipython().magic('matplotlib inline')
#let's get a visual representation of the correlations calculated
#above so we know what we can focus on

combined.corr()["sat_score"][survey_fields].plot.bar(title = "Correlation")


# hihgest correlation seem to be: rr_s, N_s, N_t, N_p, saf_t_11,
# saf_s_11, aca_s_11, saf_tot_11
# 
# r_ss = Student Response Rate
# 
# N__s = Number of student respondents (does not seem relevant)
# 
# N_t = Number of teacher respondents (does not seem relevant)
# 
# N_p = Number of parent respondents (does not seem relevant)
# 
# saf_t_11 = Safety and Respect score based on teacher responses
# (seems highly relevant)
# saf_s_11 = Safety and Respect score based on student responses
# (seems highly relevant)
# aca_s_11 = Academic expectations score based on student responses
# (seems highly relevant)
# saf_tot_11 = Safety and Respect total score 
# (seems highly relevant)

# In[12]:


#Makes a scatter plot of the saf_s_11 column vs. the sat_score in combined.
plt.title("SAT scores vs. Student School Safety Rating")
plt.scatter(combined["saf_s_11"], combined["sat_score"])


# In[13]:


avg_safety_by_dist = combined["saf_s_11"].groupby(combined["school_dist"]).agg(numpy.mean)

from mpl_toolkits.basemap import Basemap

#you must use agg(numpy.mean) in order to apply the groupby function,
#otherwise you are simply reorganizing the list (I think?)
distlon = combined.groupby("school_dist")["lon"].agg(numpy.mean)
distlat = combined.groupby("school_dist")["lat"].agg(numpy.mean)

m = Basemap(projection = 'merc', 
            #lower left corner latitude set to NYC area
            llcrnrlat = 40.496044, 
            urcrnrlat = 40.915256, 
            llcrnrlon = -74.25573,
            urcrnrlon = -73.700272,
            resolution = 'h')

plt.title("Safety by School District")
m.drawmapboundary(fill_color = '#85A6D9')
m.drawcoastlines(color='#6D5F47', linewidth = .4)
m.drawrivers(color='#85A6D9',linewidth = .4)
m.fillcontinents(color='white',lake_color='#85A6D9')

#lat and lon must be given as lists
longitudes = distlon.tolist()
latitudes = distlat.tolist()

m.scatter(x = longitudes, y = latitudes, s = 20, zorder = 2, latlon = True,
         c = avg_safety_by_dist, cmap = 'summer')

plt.show()


# In[14]:


race_cols = ['white_per', 'asian_per', 'black_per', 'hispanic_per']
plt.title("SAT Score Correlation with diff. Races")
combined.corr()["sat_score"][race_cols].plot.bar()


# It's fairly well known that test prep is strongly biased towards white and asian students but the graph above supports the argument more strongly than I would have expected.

# In[18]:


high_hispanic = combined["hispanic_per"][combined["hispanic_per"] > 95] 
plt.title("Hispanic Percent vs. SAT scores")
plt.scatter(combined["hispanic_per"], combined["sat_score"])

print("The following schools have a 95%+ hispanic student body\n")
print(combined["SCHOOL NAME"][combined["hispanic_per"] > 95] )


#     As expected, we can see that super high hispanic percentage schools are primarily from economically disadvantageous backgrounds. From our quick research, we found that a majority of the schools have 75% or more of their student body coming from economically disadvantageous backgrounds.

# In[25]:


print(combined["SCHOOL NAME"][(combined["hispanic_per"] <10) & (combined["sat_score"]>1800)] )


# The most selective schools based on SAT scores tend to be schools focused on STEM fields. They will have higher test scores because they are highly selective.

# In[32]:


genders = ["male_per", "female_per"]

combined.corr()["sat_score"][genders].plot.bar(title = "Correlation of SAT scores and gender")


# In[43]:


print("Percent female composition vs. SAT Scores")
plt.scatter(x = combined['female_per'], y =combined['sat_score'])

plt.show()
print(combined['SCHOOL NAME'][(combined['female_per'] > 60) & (combined['sat_score'] > 1700)])


# It is well documented that females tend to outperform males on standardized testing. 
# 
# The scatter plot above seems to be bi-modal but it is difficult to predict whether there is a correlation on average scores or merely a higher spread of SAT scores based on the composition of students. However, regardless of how we look at the data, the only schools that have an average test score of 1600 or higher are those with between 35% and 80% women. Perhaps the genders function better when there is more diversity in genders at the school.
# 
# Finally, most of the top performing schools with high SAT scores (1700+) and high probability of females (60%+) are listed above. Most of these schools are public college prep high schools.

# In[61]:


combined["ap_percent"] = combined["AP Test Takers "]/combined["total_enrollment"]
plt.title = "Percent AP Students vs. SAT Score"
plt.scatter(x = combined['ap_percent'], y=combined['sat_score'])


# There seems to be a very strange correlation between percentage of AP students and SAT scores - it almost seems as there is a strong positive correlation until you get to around 60%, at which point you no longer find moderately high SAT scores (anything above 1300). It could be very interesting to investigate why this is - could it be administrators pushing students too hard to perform well by "requiring" that nearly every student is an SAT student? Or was there some data glitch that gave us a perfectly straight line of scores around 1250? Is this score used as a placeholder that was forgotten? All interesting questions to look into.
