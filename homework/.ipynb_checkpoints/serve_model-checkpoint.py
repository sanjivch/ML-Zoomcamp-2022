
from flask import Flask, request, jsonify
import pickle


model_file = 'model1.bin'
dv_file = 'dv.bin'


with open(model_file, 'rb') as f:
    model = pickle.load(f)
    
with open(dv_file, 'rb') as f:
    dict_vect = pickle.load(f)

app = Flask('credit_card')

@app.route('/predict', methods=['POST'])
def predict():
    
    client = request.get_json()
    X_test = dict_vect.transform(client)
    y_pred_proba = model.predict_proba(X_test)[:,1]
    result = {
        'card_approved_probability': float(y_pred_proba)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

