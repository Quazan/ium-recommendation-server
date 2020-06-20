# -*- coding: utf-8 -*-
"""IUM_content_based.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Sk9_BJ9LVqP_VXe1TrS3sePKfb6LViIe
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

df = pd.read_json("/content/products.jsonl", lines=True)
df.describe

df = df.drop(df[df.price > 1000000].index)
df = df.drop(df[df.price <= 0].index)
df

#df["category_path"] = df["category_path"].str.split(";", n=10)
df["category_path"] = df["category_path"].str.replace(" ", "_")
df["category_path"] = df["category_path"].str.replace(";", " ")
for index, row in df.iterrows():
    df['category_path'][index] = row['category_path'] + " " + str(row['product_name'])
df['category_path'][0]

df[['category_path', 'price']]

count = CountVectorizer()
count_matrix = count.fit_transform(df['category_path'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df.index)

max_price = -1
min_price = 999999999999

for i, row in enumerate(cosine_sim):
  idxi = list(df.index)[i]
  max_price = max(max_price, df['price'][idxi])
  min_price = min(min_price, df['price'][idxi])

zmienna = 0.15

diff_table = []

for i, row in enumerate(cosine_sim):
  diff_table.append([])
  for j, elem in enumerate(row):
    idxi = list(df.index)[i]
    idxj = list(df.index)[j]
    diff = abs(df['price'][idxi] - df['price'][idxj])
    if diff != 0:
      diff_table[i].append(1/diff)
    else:
      diff_table[i].append(max_price-min_price)


diff_table = normalize(diff_table, norm='max')

for i, row in enumerate(cosine_sim):
  for j, elem in enumerate(row):
    idxi = list(df.index)[i]
    idxj = list(df.index)[j]
    cosine_sim[i][j] = diff_table[i][j] * zmienna + elem * ( 1 - zmienna)

cosine_sim

def recommendations(id, cosine_sim = cosine_sim):
    
    recommended_products = []
    
    idx = indices[indices == id].index[0]

    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    top_10_indexes = list(score_series.iloc[0:10].index)
    
    for i in top_10_indexes:
        recommended_products.append(list(df.index)[i])
    print(score_series[0:10])
    return recommended_products

rec = 304
l = recommendations(rec)
print(df['product_name'][rec])
print(l)
for e in l:
  print(df['product_name'][e])

