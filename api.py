from flask import Flask
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from PIL import Image, ImageOps
from keras.models import load_model
import numpy as np
import tensorflow as tf


app = Flask(__name__)
api = Api(app)


capd_parser = reqparse.RequestParser()
capd_parser.add_argument('dialysate', type=FileStorage, location='files', required=True, help="Upload gambar hasil buangan!")


class CAPDDetection(Resource):
  def get(self):
    return { "message": "Hello World!" }

  def post(self):
    args = capd_parser.parse_args()
    file = args['dialysate']
    image = Image.open(file)

    size = (224, 224)
    resized_image = ImageOps.fit(image, size, Image.ANTIALIAS)
    
    image_array = np.asarray(resized_image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = capd_model.predict(data)

    prediction_max = np.max(prediction[0])
    prediction_index = np.argmax(prediction[0])

    return { "label": capd_labels[prediction_index], "accuracy": str(prediction_max) }


api.add_resource(CAPDDetection, '/api')


if __name__ == "__main__":
  with tf.device('/cpu:0'):
    capd_model = load_model("./model/model.h5", compile=False)

  with open("./model/labels.txt") as file:
    capd_labels = file.read().splitlines()

  app.run(debug=True)
