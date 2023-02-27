import numpy as np
from flask import Flask, request, jsonify, render_template
import requests
import pickle
import pandas as pd
app = Flask(__name__) #initialising the flask app
import requests
# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "weFz5PmYPnMb7Xrzl6G4DL5HzxipPiLyvIa51plUJ72s"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]
header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
# NOTE: manually define and pass the array(s) of values to be scored in the next line
filepath="model_movies.pkl"
model=pickle.load(open(filepath,'rb'))#loading the saved model
scalar=pickle.load(open("scalar_movies.pkl","rb"))#loading the saved scalar file
@app.route('/')
def home():
    return render_template('Demo2.html')
@app.route('/y_predict',methods=['POST'])

def y_predict():    
    '''
    For rendering results on HTML 
    '''
    input_feature=[float(x) for x in request.form.values()]  
    features_values=[np.array(input_feature)]
    feature_name=['budget','genres','popularity','runtime','vote_average','vote_count','director','release_month','release_DOW']
    payload_scoring = {"input_data": [{"fields": ['budget','genres','popularity','runtime','vote_average','vote_count','director','release_month','release_DOW'], "values": feature_name }]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/19a32815-0863-4bfc-9781-2b843c2ab6ea/predictions?version=2023-02-10', json=payload_scoring,
     headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    predict = response_scoring.json()
    x_df=pd.DataFrame(features_values,columns=feature_name)
    x=scalar.transform(x_df)
     # predictions using the loaded model file
    prediction=model.predict(x)  
    print("Prediction is:",prediction)
    return render_template("resultnew.html",prediction_text=prediction[0])
if __name__ == "__main__":
    app.run(debug=False)
