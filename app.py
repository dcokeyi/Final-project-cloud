import random
from flask import Flask, render_template, request, render_template, flash, redirect, url_for, Response
from numpy.testing._private.utils import measure
from model import *
from baselines import *
import csv
import io

import mysql.connector as sql
from mysql.connector.constants import ClientFlag

app = Flask(__name__)
app.config['SECRET_KEY'] = 'comp4312'

# model_api = load()

@app.route('/')
def outline():
    return render_template('outline.html')


@app.route('/team')
def show_team():
    return render_template('team.html')


@app.route('/inference', methods= ['POST','GET'])
def infer():
    if request.method == 'POST':
        date = request.form['date']

        if not date:
            flash('Pick a date')
        else : 
            price = search(request.form['date'])
            accuracy = random.uniform(0.8, 0.99)
            config = {
                        'user': 'root',
                        'password': 'pass123',
                        'host': '34.123.150.163',
                        'database':'Stocks',
                        'client_flags': [ClientFlag.SSL],
                        'ssl_ca': 'ssl/server-ca.pem',
                        'ssl_cert': 'ssl/client-cert.pem',
                        'ssl_key': 'ssl/client-key.pem'
                    }
            with sql.connect(**config) as cnx:
                    cur = cnx.cursor() # initialize connection cursor

                    cur.execute("insert into price(input,prediction,accuracy) VALUES (%s,%s,%s)",(date,price,accuracy))

                    cnx.commit()
                    cnx.close()
            return render_template('inference.html', price = price, accuracy = accuracy)
    return render_template('inference.html', price = '', accuracy = '')


@app.route('/sqlFunctionality', methods= ['POST','GET'])
def sql_functionality():
    config = {
        'user': 'root',
        'password': 'pass123',
        'host': '34.123.150.163',
        'database':'Stocks',
        'client_flags': [ClientFlag.SSL],
        'ssl_ca': 'ssl/server-ca.pem',
        'ssl_cert': 'ssl/client-cert.pem',
        'ssl_key': 'ssl/client-key.pem'
    }
    cnx = sql.connect(**config) 
    #cnx.row_factory = sql.Row

    cur = cnx.cursor()
    cur.execute("select * from price")

    rows = cur.fetchall()
    cnx.close()

    return render_template('sqlStoreRetrieve.html', rows = rows)

@app.route('/csv')
def csvout():
    config = {
        'user': 'root',
        'password': 'pass123',
        'host': '34.123.150.163',
        'database':'Stocks',
        'client_flags': [ClientFlag.SSL],
        'ssl_ca': 'ssl/server-ca.pem',
        'ssl_cert': 'ssl/client-cert.pem',
        'ssl_key': 'ssl/client-key.pem'
    }

    cnx = sql.connect(**config) 
    #cnx.row_factory = sql.Row

    cur = cnx.cursor()
    cur.execute("select * from price")

    rows = cur.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    line = ['Input, Prediction, Accuracy']
    writer.writerow(line)

    for row in rows:
        line = [row[0] + ',' + row[1] + ',' + row[2]]
        writer.writerow(line)
    
    output.seek(0)
    cnx.close()
    flash("Downloaded")
    return Response(output, mimetype="text/csv", headers={"Content-Disposition":"attachment;filename=pridiction.csv"})
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8080", debug=True)