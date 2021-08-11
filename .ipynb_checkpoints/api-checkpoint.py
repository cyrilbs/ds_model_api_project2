from flask import Flask
from flask import jsonify
from flask import abort
from flask import make_response
from flask import request, abort

import pandas as pd
import json

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import NLTKWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

api = Flask(__name__)

# ---------------------------------------------------------------------------------------
# functions

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stop_words.update(["'ve", "", "'ll", "'s", ".", ",", "?", "!", "(", ")", "..", "'m", "n", "u"])
    tokenizer = NLTKWordTokenizer()
    
    text = text.lower()
    
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    
    return ' '.join(tokens)

# ---------------------------------------------------------------------------------------
# endpoints

def isAuthorized(username, password):
     return len(allBranchesDfCredentials[(allBranchesDfCredentials.username==username) & (allBranchesDfCredentials.password==int(password))].index)

@api.route('/all_branches/score', methods=['GET'])
def getAllBranchesScore():
     return jsonify({"score": allBranchesScore})

# localhost:5000/all_branches/sentiment?username=Mara&password=9820&sentence=good
@api.route('/all_branches/sentiment', methods=['GET'])
def getAllBranchesSentiment():
    username = request.args.get('username')
    password = request.args.get('password')
    sentence = request.args.get('sentence')
    if (username and password and sentence):
        if (isAuthorized(username, password)):
            if len(allBranchesDfCredentials[(allBranchesDfCredentials.username==username) & (allBranchesDfCredentials.password==int(password)) & (allBranchesDfCredentials.all_branches==1)]):
                prediction = allBranchesModel.predict(allBranchesVectorizer.transform([sentence])) # returns numpy array
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

reviewFileName = "DisneylandReviews.csv"
df = pd.read_csv(reviewFileName, encoding='cp1252')

# all branches model

print("processing all branches model ...")

allBranchesDf = df.drop(['Review_ID', 'Year_Month', 'Reviewer_Location', 'Branch'], axis=1)
allBranchesDf['Review_Text'] = allBranchesDf['Review_Text'].apply(preprocess_text)

features = allBranchesDf['Review_Text']
target = allBranchesDf['Rating']

X_train, X_test, y_train, y_test = train_test_split(features, target)

allBranchesVectorizer = CountVectorizer(max_features=2000)
X_train_cv = allBranchesVectorizer.fit_transform(X_train)
X_test_cv = allBranchesVectorizer.transform(X_test)

# allBranchesModel = RandomForestClassifier(max_depth=3, n_estimators=100)
allBranchesModel = LogisticRegression()
# allBranchesModel = DecisionTreeClassifier(max_depth=8)

allBranchesModel.fit(X_train_cv, y_train)

allBranchesScore = allBranchesModel.score(X_test_cv, y_test)

print("score:", allBranchesScore)

print("processing per branche model ...")
print("TBD")

# main flask entry

if __name__ == '__main__':
    api.run(host="0.0.0.0", port=5000)

