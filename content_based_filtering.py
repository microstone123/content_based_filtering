# imports and display settings

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

##############################################
# 1. Generating the TF-IDF Matrix
##############################################

# uploading data
# https://www.kaggle.com/rounakbanik/the-movies-dataset

df = pd.read_csv("datasets/movies_metadata.csv", low_memory=False)

def df_info(dataframe):
    print("             Take a Look at the Dataset            ")
    print("---------------------------------------------------")
    print(f"First 10 Observations: \n{dataframe.head(10)}")
    print("---------------------------------------------------")
    print(f"Last 10 Observations: \n{dataframe.tail(10)}")
    print("---------------------------------------------------")
    print(f"Dataframe Columns: \n{dataframe.columns}")
    print("---------------------------------------------------")
    print(f"Descriptive Statistics: \n{dataframe.describe().T}")
    print("---------------------------------------------------")
    print(f"NaN: \n{dataframe.isnull().sum()}")
    print("---------------------------------------------------")
    print(f"Variable Types: \n{dataframe.dtypes}")
    print("---------------------------------------------------")
    print(f"Number of Observations: \n{dataframe.shape[0]}")
    print(f"Number of Variables: \n{dataframe.shape[1]}")

df_info(df)

df["overview"].head()

# we want to eliminate frequently used words in english such as the, and, on
tfidf = TfidfVectorizer(stop_words="english")

# Replacing missing information in the overview variable with " "
df["overview"] = df["overview"].fillna(" ")

# eliminating frequently used words in "overview" varibale
tfidf_matrix = tfidf.fit_transform(df["overview"])

tfidf_matrix.shape
# 45466 : overviews
# 75827 : unique words
# tf-idf scores exist at the intersection of these two

# specifying the types of values that make up the matrix as float32
tfidf_matrix = tfidf_matrix.astype(np.float32)

# while creating the cosine similarity matirx, I get ArrayMemoryError, limiting the matrix will solve this error
tfidf_matrix = tfidf_matrix[:15000, :15000]

# convert matrix to array
tfidf_matrix.toarray()

##############################################
# 2.Creating the Cosine Similarity MAtrix
##############################################

# 查找每部电影与其他电影的相似度值
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape

# 第一个索引中的电影与所有其他电影的相似度分数
cosine_sim[1]


##############################################
# 3. 根据相似性提出推荐
##############################################

# 关于哪部电影在哪个索引中的信息
indices = pd.Series(df.index, index=df["title"])

indices.index.value_counts()
# 可见标题中存在复用。

# 保留这些倍数中的一个并消除其余的
# 在最近的日期取这些倍数中最近的一个。
indices = indices[~indices.index.duplicated(keep="last")]

movie_index = indices["Toy Story"]
# 限制矩阵，用户选择仍在有限矩阵中的电影

# 福尔摩斯与其他电影的相似度得分
similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])

# 获取 10 部最相似电影的索引
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
# 第 0 个索引包含电影本身，从第 1 个索引中选择相似电影。

# 在这些索引中获取电影标题
df["title"].iloc[movie_indices]
