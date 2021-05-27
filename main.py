from flask import Flask, request
from flask_cors import CORS
import bcrypt
import base64
import io
from imageio import imread
from PIL import Image, ImageOps
import tensorflow
import numpy as np

app = Flask(__name__)
CORS(app)

covid_model = tensorflow.keras.models.load_model('covid_model.h5')
lung_model = tensorflow.keras.models.load_model('InceptionResNetV2-lung-and-colon-tumor-99.83.h5')

################################################################
#                                                              #
#                       FUNCTIONS                              #
#                                                              #
################################################################

def encrypt(password):
    hashedPass = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    encoding = 'utf-8'
    normHashedPass = str(hashedPass, encoding)
    return {"hashPass": normHashedPass, "status": "success"}

def verify(hashPassword, userPassword):
    encHashPassword = hashPassword.encode("utf-8")
    encUserPassword = userPassword.encode("utf-8")
    if bcrypt.checkpw(encUserPassword, encHashPassword):
        return {"status":"success"}
    else:
        return {"status":"failed"}

def predict_covid(img_string):
    image = imread(io.BytesIO(base64.b64decode(img_string)))

    IMG_WIDTH = 224
    IMG_HEIGHT = 224

    img_data_array=[]
            
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH,IMG_HEIGHT))
    image=np.array(image)
    image = image.astype('float32')
    image /= 255
    print(image.shape)
    image = image.reshape(1, 224, 224, 3)
    img_data_array.append(image)
    
    pred = covid_model.predict(img_data_array,batch_size=32)

    final = pred[0][0]

    if final < 0.5:
        return {"status": "success", "prediction": "benign", "confidence_score": final, "base64": img_string}
    else:
        return {"status": "success", "prediction": "malignant", "confidence_score": final, "base64": img_string}

def predict_lung(img_string):
    image = imread(io.BytesIO(base64.b64decode(img_string)))

    IMG_WIDTH = 128
    IMG_HEIGHT = 128

    image = ImageOps.fit(image, (IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
    final_data = data = np.ndarray(shape=(1, IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.float32)
    data = np.asarray(image)
    data = data.astype("float32")
    data /= 127.5
    data-=1
    final_data[0] = data
    diseases = ["colon_aca", "colon_benign", "lung_aca", "lung_benign", "lung_scc"]
    
    pred = list(lung_model.predict(final_data)[0])

    final = max(pred)
    
    if final < 0.5:
        return {"status": "success", "disease_type": diseases[pred.index(max(pred))], 'prediction': "benign", "confidence_score": final, "base64": img_string}
    else:
        return {"status": "success", "disease_type": diseases[pred.index(max(pred))], 'prediction': "malignant", "confidence_score": final, "base64": img_string}

################################################################
#                                                              #
#                       ENDPOINTS                              #
#                                                              #
################################################################

@app.route("/")
def hello():
    return "Hello, Hygeia"

@app.route('/register/encrypt', methods=["GET", "POST"])
def encryption_endpoint():
    data = request.json
    return encrypt(data["password"])

@app.route('/login/verify', methods=["GET", "POST"])
def verfication_endpoint():
    data = request.json
    return verify(data["hashPassword"], data["password"])


@app.route('/predict', methods=["GET", "POST"])
def predict_endpoint():
    data = request.json
    if (data["covid"] == "True"):
        return predict_covid(data["img"])
    else:
        return predict_lung(data["img"])

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)