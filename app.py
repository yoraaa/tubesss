from flask import Flask,render_template,request,redirect
import pickle
from flask.scaffold import _matching_loader_thinks_module_is_package
import sklearn
import numpy as np

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def index():
    if request.method=='POST':
        
        with open('knn_pickle','rb') as r:
            model = pickle.load(r)
            
        peserta = int(request.form['peserta'])
        rombongan = int(request.form['rombongan'])
        guru = int(request.form['guru'])
        pegawai = int(request.form['pegawai'])
        kelas = int(request.form['kelas'])
        lab = int(request.form['lab'])
        perpus = int(request.form['perpus'])
        
        datas = np.array((peserta,rombongan,guru,pegawai,kelas,lab,perpus))
        datas = np.reshape(datas, (1, -1))
        
        isStatus = model.predict(datas)
        
        
        
        return render_template('hasil.html',finalData=isStatus)

    else:
        return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug=True)