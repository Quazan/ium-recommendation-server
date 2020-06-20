from flask import Flask, jsonify, request
from collaborativeFiltering import CollaborativeFilteringModel
from contentBased import ContentBasedModel
import pickle

app = Flask(__name__)


coll = open('resources\\collaborativeFiltering.txt', 'rb')
collaborative = pickle.load(coll)
coll.close()

con = open('resources\\contentBased.txt', 'rb')
content = pickle.load(con)
con.close()


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict')
def predict():
    userId = int(request.args.get('userId'))
    productId = int(request.args.get('productId'))

    if userId % 2 == 0:
        pred = content.predict(productId)
        mod = 'content based'
    else:
        pred = collaborative.predict(userId)
        mod = 'user based'

    return jsonify(user=userId,
                   product=productId,
                   prediction=pred,
                   model=mod)


@app.route('/collaborative')
def predictCollaborative():
    userId = int(request.args.get('userId'))
    return jsonify(user=userId,
                   prediction=collaborative.predict(userId))


@app.route('/content')
def predictionContent():
    productId = int(request.args.get('productId'))
    return jsonify(product=productId,
                   prediction=content.predict(productId))


if __name__ == '__main__':
    app.run()
