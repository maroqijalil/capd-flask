from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from PIL import Image, ImageOps
from tensorflow import keras
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
    normalized_image_array = (image_array.astype(np.float32) / 255.0)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = image_array
    preds = capd_model.predict(data).flatten()

    preds_proba = tf.nn.sigmoid(preds)
    preds = tf.where(preds_proba < 0.5, 0, 1)
    preds_result = preds.numpy().tolist()

    labels_list = list(capd_labels)

    return { "label": str(labels_list[int(preds_result[0])]), "accuracy": str(preds_proba.numpy().tolist()) }


api.add_resource(CAPDDetection, '/api')


if __name__ == "__main__":
  with tf.device('/cpu:0'):
    capd_model = keras.models.load_model("./model/best_model.h5", compile=False)

  with open("./model/labels.txt") as file:
    capd_labels = file.read().splitlines()

  app.run(host='0.0.0.0', debug=True, port=5001)
