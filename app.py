import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.errorhandler(404)
def pageNotFound(e):
    return render_template('404.html'), 404

@app.errorhandler(405)
def methodNotFound(e1):
    return render_template('405.html'), 405


@app.errorhandler(500)
def incorrectinput(error):
    return render_template('500.html'), 500

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    int_features = [[float(x) for x in request.form.values()]]
    final = np.array(int_features)

    prediction = model.predict(final)
    output = prediction[0]
    if output==0:
        s= 'Negative'
    elif output==1:
        s = 'Positive'
    proba = model.predict_proba(final)

    prob1 = proba[0][1]*100
    if prob1>int(70):
       a="High"
    elif int(30)<prob1<=int(70):
       a="Medium"
    else:
       a="Low"
    return render_template('result.html',
                               pred='Test Result : {}'.format(s)
                           ,pred1='Percentage of risk  : {:.2f}%'.format(prob1)
                           ,pred2='Risk Level : {}'.format(a))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    final = np.array([list(data.values())])
    prediction = model.predict_proba(final)
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)