import read
import collections

df = read.load_data()

headlines = df['headline']
#join all headlines together
string = ""

for i in headlines:
    string = string + " " + str.lower(str(i))
print("Successfully joined headlines into one string")

#split strings
headline_words = string.split()
print("Successfully split string into individual words")

#count occurances
c = collections.Counter(headline_words)
print(c.most_common(100))
print("Successfully print most common 100 words")