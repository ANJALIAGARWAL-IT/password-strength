# Importing essential libraries
from flask import Flask, render_template, request
import pickle

def character(input):
    char = []
    for i in input:
        char.append(i)
    return char

# Load the Random Forest model and CountVectorizer object from disk
filename = 'password-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl', 'rb'))
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['password']
        X=character([message])
        vect = cv.transform(X).toarray()
        #vect=cv.fit_transform(X)
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)
       

if __name__ == '__main__':
    app.run(debug=True)
