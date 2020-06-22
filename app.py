from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from collaborativeFiltering import CollaborativeFilteringModel
from contentBased import ContentBasedModel
import pickle
import os
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

### swagger specific ###
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "IUM-recommendation-server"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
### end swagger specific ###

coll = open('collaborativeFiltering.txt', 'rb')
collaborative = pickle.load(coll)
coll.close()

con = open('contentBased.txt', 'rb')
content = pickle.load(con)
con.close()

#DATABASE_URL = os.environ['DATABASE_URL']
DATABASE_URL = 'postgres://qryefukcoptifa:8d38e8b6b07b4427cf3183901c2fe54e71856c55e78547a8e5b79aebef44de9e@ec2-79-125-26-232.eu-west-1.compute.amazonaws.com:5432/d1bhsiu68fpo53'
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class Logs(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    mode = db.Column(db.String, nullable=False)
    prediction = db.Column(db.String, nullable=False)
    product_id = db.Column(db.Integer, nullable=True)
    user_id = db.Column(db.Integer, nullable=True)


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

    data = Logs(user_id=userId, product_id=productId, mode=mod, prediction=str(pred))
    db.session.add(data)
    db.session.commit()
    return jsonify(user=userId,
                   product=productId,
                   prediction=pred,
                   model=mod)


@app.route('/collaborative')
def predictCollaborative():
    userId = int(request.args.get('userId'))
    pred = collaborative.predict(userId)
    data = Logs(user_id=userId, mode='user based', prediction=str(pred))
    db.session.add(data)
    db.session.commit()
    return jsonify(user=userId,
                   prediction=pred)


@app.route('/content')
def predictionContent():
    productId = int(request.args.get('productId'))
    pred = content.predict(productId)
    data = Logs(product_id=productId, mode='content based', prediction=str(pred))
    db.session.add(data)
    db.session.commit()
    return jsonify(product=productId,
                   prediction=pred)


if __name__ == '__main__':
    app.run()
