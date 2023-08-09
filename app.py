import pickle

from flask import Flask, render_template, request, url_for

import numpy as np
from flask import jsonify
# initiate flask
app = Flask(__name__)

o2 =  {'Obesity_Type_I': 0,
  'Obesity_Type_III': 1,
  'Obesity_Type_II': 2,
  'Overweight_Level_I': 3,
  'Overweight_Level_II': 4,
  'Normal_Weight': 5,
  'Insufficient_Weight': 6}



with open('std.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('RandomForest.pkl', 'rb') as file:
    crop_recommendation_model = pickle.load(file)

with open('lir.pkl', 'rb') as file:
    lir = pickle.load(file)

with open('std2.pkl', 'rb') as file:
    scaler2 = pickle.load(file)


with open('rfc2.pkl', 'rb') as file:
    rf2 = pickle.load(file)

@app.route('/pred1/',methods = ['POST'])
def home() :

    if request.method == 'POST':
        data = request.get_json()
        print(data)
        l = data["data"]
        Inp = []
        for x in l :
            Inp.append(int(x))
        X = scaler.transform([Inp])

        out = lir.predict(X)[0]
        if out < 0 :
            out = 0
        d = {"Prediction" : out*100}

    return jsonify(d)

@app.route('/pred2/',methods = ['POST'])
def home1() :

    if request.method == 'POST':
        data = request.get_json()
        print(data)
        l = data["data"]
        Inp = []
        for x in l :
            Inp.append(int(x))
        X = scaler2.transform([Inp])

        out = rf2.predict(X)[0]

        d = {"Prediction" : list(o2.keys())[out]}

    return jsonify(d)

def weather_fetch(city_name):
    api_key = "9d7cde1f6d07ec55650544be1631307e"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

@app.route('/crop_predict')
def home2(){
    N = 55
    P = 56
    K = 57
    ph = 7.0
    rainfall = 177
    state = "Haryana"
    city = "Hisar"
    temperature, humidity = weather_fetch(city)
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    my_prediction = crop_recommendation_model.predict(data)
    final_prediction = my_prediction[0]
    print("Crop to be Grown : ", final_prediction)
    return jsonify(final_prediction)
}

if __name__ == '__main__':
    app.run(debug = True)
