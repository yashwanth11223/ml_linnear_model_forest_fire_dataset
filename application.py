from flask import Flask,render_template,request
import pickle
import numpy as np
application=Flask(__name__)

app=application

linear=pickle.load(open('model/linear.pkl','rb'))
scaler=pickle.load(open('model/scaler.pkl','rb'))
@app.route("/")
def Home():
    return render_template("home.html")
@app.route("/pred",methods=['GET',"POST"])
def pred():
    if request.method=='POST':
        day=int(request.form.get('day'))
        month=int(request.form.get('month'))
        year=int(request.form.get('year'))
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC')) 
        DC=float(request.form.get('DC')) 
        ISI=float(request.form.get('ISI')) 
        BUI=float(request.form.get('BUI')) 
        Classes=float(request.form.get('Classes')) 
        Region=float(request.form.get('Region')) 

        new_s=scaler.transform([[day,month,year,Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Classes,Region]])
        res=linear.predict(new_s)
        return render_template('home.html',res=res[0])
if __name__=="__main__":
    app.run(host="0.0.0.0")