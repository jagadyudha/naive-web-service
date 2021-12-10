from flask import Flask
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import jsonify, make_response
from sklearn.naive_bayes import GaussianNB
from flask import request

app = Flask(__name__)


@app.route('/')
def home():
    return 'Hallo'

@app.route('/naive', methods=['POST'])
def naive():
    data = [request.json]
    data_training = pd.read_excel("data_training.xlsx")
    enc = LabelEncoder()
    data_training['Gender'] = enc.fit_transform(data_training['Gender'].values)
    data_training['Ever_Married'] = enc.fit_transform(data_training['Ever_Married'].values)
    data_training['Age'] = enc.fit_transform(data_training['Age'].values)
    data_training['Graduated'] = enc.fit_transform(data_training['Graduated'].values)
    data_training['Profession'] = enc.fit_transform(data_training['Profession'].values)
    data_training['Spending_Score'] = enc.fit_transform(data_training['Spending_Score'].values)
    x = data_training.drop(["Segmentation"], axis=1)
    y = data_training["Segmentation"]
    data_test = pd.DataFrame(data)
    x_test = data_test.drop(["Segmentation"], axis=1)
    y_test = data_test["Segmentation"]
    modelnb = GaussianNB()
    nbtrain = modelnb.fit(x, y)
    Y_predict = nbtrain.predict(x_test)
    result = data + [{"prediction":Y_predict[0]}]
    return make_response(jsonify(result))