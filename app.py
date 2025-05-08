from flask import Flask, jsonify, request, render_template, redirect, url_for
from prediction_model import PredictionModel
import pandas as pd
from random import randrange
from forms import OriginalTextForm
import sqlite3
import google.generativeai as genai


# Load Gemini AI model
genai.configure(api_key='AIzaSyB2zvehKi1RBJ4dyWvg1i1-YiG_RAopFtg')
gemini_model = genai.GenerativeModel('gemini-pro')
chat = gemini_model.start_chat(history=[])


app = Flask(__name__)

app.config['SECRET_KEY'] = '4c99e0361905b9f941f17729187afdb9'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return redirect(url_for('home'))

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route("/home", methods=['POST', 'GET'])
def home():
    form = OriginalTextForm() 

    if form.generate.data:
        data = pd.read_csv("random_dataset.csv")
        index = randrange(0, len(data)-1, 1)
        original_text = data.loc[index].text
        form.original_text.data = str(original_text)
        return render_template('home.html', form=form, output=False)

    elif form.predict.data:
        if len(str(form.original_text.data)) > 10:
            print(f"\n\n\n input is : {str(form.original_text.data)} \n\n\n")
            model = PredictionModel(form.original_text.data)
            # print(f"\n\n\n{model.predict()}\n\n\n")
            output=model.predict()
            org=output['original']
            print(f"\n\n\n input were : {org} \n\n\n")
            gemini_response = chat.send_message(org+" this is the News is need to examine wheather it is Real or Fake so give me the single word reply wheather it is Real or Fake ")
            res = gemini_response.text
            print(f"ans is : ------- >  \n\n\n {res} \n\n\n\n")
            # recommendatoin = recommendatoin.replace("```html", "")
            # recommendatoin = recommendatoin.replace("```", "")
            return render_template('home.html', form=form, output=output,res=res)

    return render_template('home.html', form=form, output=False)


@app.route('/predict/<original_text>', methods=['POST', 'GET'])
def predict(original_text):
    #text = 'CAIRO (Reuters) - Three police officers were killed and eight others injured in a shoot-out during a raid on a suspected militant hideout in Giza, southwest of the Egyptian capital, two security sources said on Friday. The sources said authorities were following a lead to an apartment thought to house eight suspected members of Hasm, a group which has claimed several attacks around the capital targeting judges and policemen since last year. The suspected militants fled after the exchange of fire there, the sources said. Egypt accuses Hasm of being a militant wing of the Muslim Brotherhood, an Islamist group it outlawed in 2013. The Muslim Brotherhood denies this. An Islamist insurgency in the Sinai peninsula has grown since the military overthrew President Mohamed Mursi of the Muslim Brotherhood in mid-2013 following mass protests against his rule. The militant group staging the insurgency pledged allegiance to Islamic State in 2014. It is blamed for the killing of hundreds of soldiers and policemen and has started to target other areas, including Egypt s Christian Copts. ' 
    model = PredictionModel(original_text)
    return jsonify(model.predict())


@app.route('/random', methods=['GET'])
def random():
    data = pd.read_csv("random_dataset.csv")
    index = randrange(0, len(data)-1, 1)
    return jsonify({'title': data.loc[index].title, 'text': data.loc[index].text, 'label': str(data.loc[index].label)})

@app.route("/logout")
def logout():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
