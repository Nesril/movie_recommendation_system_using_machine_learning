import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ast#to make access over objects 
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity


movie_data=pd.read_csv("movies.csv")
credit_data=pd.read_csv("credits.csv")
#pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)
movie_data=movie_data.merge(credit_data,on="title")
#print(movie_data.columns.sort_values(ascending=True))
movie_data=movie_data[["movie_id","cast","title","overview","genres","keywords","crew"]]
movie_data=movie_data.dropna()

#we make recommandation based on genres, overview,keywords,cast and crew
def convert(obj):
  names=[]
  for i in ast.literal_eval(obj):
    names.append(i["name"])
  return names

movie_data.keywords=movie_data.keywords.apply(convert)
movie_data.genres=movie_data.genres.apply(convert)

def fetch_director(obj):
  names=[]
  for i in ast.literal_eval(obj):
    if i["job"]=="Director":
     names.append(i["name"])
  return names

movie_data.crew=movie_data.crew.apply(fetch_director)

#you should separate recomandation text
def partition_overview(text):
  new_overview=[]
  old_overview=text.split(" ")
  try:
    old_overview.remove("a")
    for i in range(0,len(old_overview),4):
        new_group = " ".join(old_overview[i:i+4])
        new_overview.append(new_group)
  except:
    for i in range(0,len(old_overview),4):
        new_group = " ".join(old_overview[i:i+4])
        new_overview.append(new_group)
  return new_overview

movie_data.overview=movie_data.overview.apply(partition_overview)

#select 4 casted peoples
def select_someCasts(obj):
  names=[]
  count=0
  for i in ast.literal_eval(obj):
    if count<3:
     names.append(i["name"])
     count=count+1
  return names

movie_data.cast=movie_data.cast.apply(select_someCasts)
movie_data.genres = movie_data["genres"].apply(lambda x:[i.replace(" ", "") for i in x])
movie_data.cast = movie_data["cast"].apply(lambda x:[i.replace(" ", "") for i in x])
movie_data.crew = movie_data["crew"].apply(lambda x:[i.replace(" ", "") for i in x])

movie_data["tags"]=movie_data["overview"]+movie_data["genres"]+movie_data["cast"]+movie_data["keywords"]+movie_data["crew"]
finalData=movie_data[["movie_id","title","tags"]]

def make_tags_string(array):
   text=""
   for i in array:
     text=text+" "+i.lower()
   return text

finalData.tags=finalData.tags.apply(lambda x:" ".join(x))


tf=TfidfVectorizer(max_features=5000,stop_words="english")#The `stop_words="english"` parameter specifies that common English words (e.g., "a", "the", "and", etc.) should be excluded from the feature set.
vecterizer=tf.fit_transform(finalData.tags).toarray()

ps=PorterStemmer()
def stemString(text):
   eachTags=[]
   for i in text:
     eachTags.append(ps.stem(i))
   return " ".join(eachTags)
finalData.tags=finalData.tags.apply(stemString)


similarity=cosine_similarity(vecterizer)

#x=sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

print(similarity.shape)
print(finalData.shape)
def recommendation(movie):
   movie_index=finalData[finalData["title"]==movie].index[0]
   print(movie_index)
   distance=similarity[movie_index]
   movie_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
   for i in movie_list:
      print(finalData.iloc[i[0]].title)
    
recommendation("Iron Man")
print("-------------------------")
recommendation("The Dark Knight Rises")