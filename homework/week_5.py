import pickle
import sklearn

model_file = 'model1.bin'
dv_file = 'dv.bin'

client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

with open(model_file, 'rb') as f:
    model = pickle.load(f)
    
with open(dv_file, 'rb') as f:
    dict_vect = pickle.load(f)
    
X = dict_vect.transform(client)

prediction = model.predict_proba(X)[0,1]
print(f"Probability of client getting a credit card: {prediction}")
