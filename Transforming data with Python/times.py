import read
import pandas as pd
import collections
import dateutil
import datetime

df = read.load_data()

def dates(i):
    y = dateutil.parser.parse(i)
    return y.hour

df['hour'] = df['submission_time'].apply(dates)
print(df['hour'].value_counts())

#It appears that the most popular times to post are between 2 and 5pm! Followed shortly by 6-8pm