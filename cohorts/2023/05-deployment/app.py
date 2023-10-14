from flask import Flask, request, jsonify
import pickle
import numpy as np

with open("homework/dv.bin", "rb") as infile:
    dv = pickle.load(infile)

with open("homework/model1.bin", "rb") as infile:
    model = pickle.load(infile)

def prepare_features(obs):
    client_data = dv.transform(obs)
    return client_data

def predict(client_data):
    preds = model.predict_proba(client_data)[0]
    pred = model.predict(client_data)[0]
    return (preds, pred)

app = Flask('credit-scoring')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    obs = request.get_json()

    client_data = prepare_features(obs)
    (preds, pred) = predict(client_data)

    data = {'score': preds.tolist(), 'class': pred}
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)