import pandas as pd
import json
import pickle

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import NLTKWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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
# main

reviewFileName = "DisneylandReviews.csv"
df = pd.read_csv(reviewFileName, encoding='cp1252')

# all branches model

print("processing all branches model ...")

allBranchesSavingFileName = 'allBranchesModel.sav'

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

print("score is", allBranchesScore)

print("saving model, test data and vectorizer in", allBranchesSavingFileName)

toSave = {
  "X_test": X_test_cv,
  "y_test": y_test,
  "model": allBranchesModel,
  "vectorizer": allBranchesVectorizer 
}

pickle.dump(toSave, open(allBranchesSavingFileName, 'wb'))

#print("processing per branche model ...")
#print("TBD")

