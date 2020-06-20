import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringModel:
    def __init__(self):
        sessions = pd.read_json('sessions.jsonl', lines=True)
        self.products = pd.read_json('products.jsonl', lines=True)

        sessionToUser = {}
        for index, row in sessions.iterrows():
            sessionToUser[row["session_id"]] = row["user_id"]

        for index, row in sessions.iterrows():
            sessions["user_id"][index] = sessionToUser[row["session_id"]]

        sessions = sessions[sessions['product_id'].notna()]

        buySessions = sessions[sessions['event_type'] == 'BUY_PRODUCT']
        sessions = sessions[sessions['event_type'] == 'VIEW_PRODUCT']

        userWatch = sessions[['user_id', 'product_id', 'session_id']].groupby(['user_id', 'product_id']).agg(['count'])
        userBuy = buySessions[['user_id', 'product_id', 'session_id']].groupby(['user_id', 'product_id']).agg(['count'])

        ld = []
        for i, r in userBuy.iterrows():
            ld.append({'user_id': r.name[0], 'product_id': r.name[1], 'count': r[0]})

        userBoughtList = pd.DataFrame(ld)

        ld = []
        for i, r in userWatch.iterrows():
            ld.append({'user_id': r.name[0], 'product_id': r.name[1], 'count': r[0]})

        self.userWatchlist = pd.DataFrame(ld)

        self.mean = self.userWatchlist.groupby(by='user_id', as_index=False)['count'].mean()

        self.userWatchlist = pd.merge(self.userWatchlist, self.mean, on='user_id')
        self.userWatchlist['count'] = self.userWatchlist['count_x'] - self.userWatchlist['count_y']

        check = self.userWatchlist.pivot(index='user_id', columns='product_id', values='count')
        self.checkBought = userBoughtList.pivot(index='user_id', columns='product_id', values='count')
        self.watchlistPivot = check.fillna(check.mean(axis=0))

        cosine = cosine_similarity(self.watchlistPivot)
        np.fill_diagonal(cosine, 0)
        self.similarityWithProduct = pd.DataFrame(cosine, index=self.watchlistPivot.index)
        self.similarityWithProduct.columns = self.watchlistPivot.index

        self.simUser30m = self.findNeighbours(self.similarityWithProduct, 30)
        self.userWatchlist = self.userWatchlist.astype({'product_id': str})
        self.productUser = self.userWatchlist.groupby(by='user_id')['product_id'].apply(lambda x: ','.join(x))

    def findNeighbours(self, df, n):
        df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
                                          .iloc[:n].index,
                                          index=['top{}'.format(i) for i in range(1, n + 1)]), axis=1)
        return df

    def predict(self, user):
        productsBoughtByUser = self.checkBought.columns[
            self.checkBought[self.checkBought.index == user].notna().any()].tolist()
        a = self.simUser30m[self.simUser30m.index == user].values
        b = a.squeeze().tolist()
        d = self.productUser[self.productUser.index.isin(b)]
        l = ','.join(d.values)
        productsSeenBySimilarUsers = l.split(',')
        productsUnderConsideration = list(
            set(productsSeenBySimilarUsers) - set(list(map(str, productsBoughtByUser))))
        productsUnderConsideration = list(map(float, productsUnderConsideration))
        score = []
        for item in productsUnderConsideration:
            c = self.watchlistPivot.loc[:, item]
            d = c[c.index.isin(b)]
            f = d[d.notnull()]
            avg_user = self.mean.loc[self.mean['user_id'] == user, 'count'].values[0]
            index = f.index.values.squeeze().tolist()
            corr = self.similarityWithProduct.loc[user, index]
            fin = pd.concat([f, corr], axis=1)
            fin.columns = ['adg_score', 'correlation']
            fin['score'] = fin.apply(lambda x: x['adg_score'] * x['correlation'], axis=1)
            nume = fin['score'].sum()
            deno = fin['correlation'].sum()
            finalScore = avg_user + (nume / deno)
            score.append(finalScore)
        data = pd.DataFrame({'product_id': productsUnderConsideration, 'score': score})
        top5recommendation = data.sort_values(by='score', ascending=False).head(5)
        top5recommendation.product_id = pd.to_numeric(top5recommendation.product_id)
        productsName = top5recommendation.merge(self.products, how='inner', on='product_id')
        productsNames = productsName.product_name.values.tolist()
        return productsNames
