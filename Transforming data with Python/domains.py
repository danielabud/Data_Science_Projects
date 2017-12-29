import read
import pandas as pd
import collections

df=read.load_data()

#count domains
domains = df['url'].value_counts()
print("Successfully counted URLs")

for name, row in domains.items():
    print("{0}: {1}".format(name, row))
    
#c = collections.Counter(df['url'])
#print(c.most_common(100))

#To remove subdomains, we could use tl