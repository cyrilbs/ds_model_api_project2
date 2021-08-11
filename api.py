from flask import Flask
from flask import jsonify
from flask import abort
from flask import make_response
from flask import request, abort

import pandas as pd
import json
import pickle

api = Flask(__name__)

# ---------------------------------------------------------------------------------------
# endpoints

def isAuthorized(username, password):
     return len(allBranchesDfCredentials[(allBranchesDfCredentials.username==username) & (allBranchesDfCredentials.password==int(password))].index)

@api.route('/all_branches/score', methods=['GET'])
def getAllBranchesScore():
     return jsonify({"score": allBranchesModel["model"].score(allBranchesModel["X_test"], allBranchesModel["y_test"])})

# localhost:5000/all_branches/sentiment?username=Mara&password=9820&sentence=good
@api.route('/all_branches/sentiment', methods=['GET'])
def getAllBranchesSentiment():
    username = request.args.get('username')
    password = request.args.get('password')
    sentence = request.args.get('sentence')
    if (username and password and sentence):
        if (isAuthorized(username, password)):
            if len(allBranchesDfCredentials[(allBranchesDfCredentials.username==username) & (allBranchesDfCredentials.password==int(password)) & (allBranchesDfCredentials.all_branches==1)]):
                prediction = allBranchesModel["model"].predict(allBranchesModel["vectorizer"].transform([sentence])) # returns numpy array
                return jsonify({"sentiment": prediction.tolist()})
            else:
                abort(403)
        else:
            abort(404)
    else:
       abort(400)

# ---------------------------------------------------------------------------------------
# exceptions

@api.errorhandler(404)
def resource_not_found(error):
    return make_response(jsonify({'error': 'Resource not found'}), 404)

@api.errorhandler(403)
def resource_not_found(error):
    return make_response(jsonify({'error': 'Not authorized'}), 403)

@api.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)

# ---------------------------------------------------------------------------------------
# main

credentialsFileName = "credentials.csv"
allBranchesDfCredentials = pd.read_csv(credentialsFileName, sep=",")

# all branches model

print("loading all branches model ...")
allBranchesSavingFileName = 'allBranchesModel.sav'
allBranchesModel = pickle.load(open(allBranchesSavingFileName, 'rb'))

print("score is", allBranchesModel["model"].score(allBranchesModel["X_test"], allBranchesModel["y_test"]))
testSentence = "this was amazing!"
print("testing sentiment on test sentence \""+testSentence+"\":", allBranchesModel["model"].predict(allBranchesModel["vectorizer"].transform([testSentence])))

# main flask entry

if __name__ == '__main__':
    api.run(host="0.0.0.0", port=5000)

