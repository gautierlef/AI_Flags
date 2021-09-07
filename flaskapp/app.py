from flask import Flask, flash, render_template, request, redirect, url_for, session
import os
import random
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    def allowed_file(testfilename):
        return '.' in testfilename and \
               testfilename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    upload_folder =  './static/media/uploads'
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
    app.config['upload_folder'] = upload_folder
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(upload_folder + '/' + filename)
            session['upload_folder'] = upload_folder
            session['filename'] = filename
            return redirect('/flag')
    return render_template('home.html')

@app.route('/flag', methods=['GET'])
def flag():
    upload_folder = session['upload_folder']
    filename = session['filename']
    path = upload_folder + '/' + filename
    test = random.randint(1, 3)
    if test == 1:
        predict = "Je pense que c'est un drapeau Basque !"
    if test == 2:
        predict = "Je pense que c'est un drapeau Belge !"
    if test == 3:
        predict = "Je pense que c'est un drapeau Français !"
    return render_template('flag.html', path=path, predict=predict)

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)