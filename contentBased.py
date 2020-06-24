import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


class ContentBasedModel:

    def __init__(self):
        self.df = pd.read_json("products.jsonl", lines=True)

        self.df = self.df.drop(self.df[self.df.price > 1000000].index)
        self.df = self.df.drop(self.df[self.df.price <= 0].index)

        self.df["category_path"] = self.df["category_path"].str.replace(" ", "_")
        self.df["category_path"] = self.df["category_path"].str.replace(";", " ")
        for index, row in self.df.iterrows():
            self.df['category_path'][index] = row['category_path'] + " " + str(row['product_name'])

        count = CountVectorizer()
        count_matrix = count.fit_transform(self.df['category_path'])

        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)

        self.indices = pd.Series(self.df.index)

        max_price = -1
        min_price = 999999999999

        for i, row in enumerate(self.cosine_sim):
            idxi = list(self.df.index)[i]
            max_price = max(max_price, self.df['price'][idxi])
            min_price = min(min_price, self.df['price'][idxi])

        zmienna = 0.15

        diff_table = []

        for i, row in enumerate(self.cosine_sim):
            diff_table.append([])
            for j, elem in enumerate(row):
                idxi = list(self.df.index)[i]
                idxj = list(self.df.index)[j]
                diff = abs(self.df['price'][idxi] - self.df['price'][idxj])
                if diff != 0:
                    diff_table[i].append(1 / diff)
                else:
                    diff_table[i].append(max_price - min_price)

        diff_table = normalize(diff_table, norm='max')

        for i, row in enumerate(self.cosine_sim):
            for j, elem in enumerate(row):
                self.cosine_sim[i][j] = diff_table[i][j] * zmienna + elem * (1 - zmienna)

    def predict(self, product_id):
        id = self.df.index[self.df['product_id'] == product_id].tolist()[0]
        recommended_products = []
        idx = self.indices[self.indices == id].index[0]
        score_series = pd.Series(self.cosine_sim[idx]).sort_values(ascending=False)
        top_10_indexes = list(score_series.iloc[0:10].index)
        for i in top_10_indexes:
            recommended_products.append(list(self.df.index)[i])

        prediction = []
        for e in recommended_products:
            if e == id:
                continue
            prediction.append(self.df['product_name'][e])

        return prediction
