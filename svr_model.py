from flask import Flask, render_template, request
from interference import MaskModel
from werkzeug import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded'
im_path = 'img.png'

# @app.route("/prediction", methods=['GET'])
def mask(): 
    model = MaskModel()
    img = model.get_prediction(im_path)
    return(img)

@app.route("/save", methods=['GET'])
def save(img): 
    return img

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
