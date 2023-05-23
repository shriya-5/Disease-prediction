import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image
import torchvision
import torch
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

#model =tf.keras.models.load_model('covid_classifier.pt',compile=False)
resnet50 = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
resnet50.fc = torch.nn.Linear(in_features=2048, out_features=3)

resnet50.load_state_dict(torch.load('C:\\Users\\SHRIYA\\Documents\\DACC\\covid_classifier1.pt'))
resnet50.eval()
print('Model loaded. Check http://127.0.0.1:5000/')

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)),                            
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )])

class_names = ['normal', 'viral', 'covid']
def predict_image_class(image_path):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image)
    image = image.unsqueeze(0)
    output = resnet50(image)[0]
    probabilities = torch.nn.Softmax(dim=0)(output)
    probabilities = probabilities.cpu().detach().numpy()
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = class_names[predicted_class_index]
    return probabilities, predicted_class_index, predicted_class_name


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        probabilities, predicted_class_index, predicted_class_name = predict_image_class(file_path)
        print(predicted_class_name)

        # x = x.reshape([64, 64]);
        class_names = ['normal', 'viral', 'covid']
        #a = preds[0]
        #ind=np.argmax(a)
        print('Prediction:', class_names[predicted_class_index])
        result=class_names[predicted_class_index]
        return result
    return None


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0',port=80)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()
