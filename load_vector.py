import array
import oracledb
import os
import onnx
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import matplotlib.pyplot as plt

#########################################################
##https://github.com/onnx/models/blob/main/validated/vision/classification/onnxrt_inference.ipynb
#########################################################
#########################################################
##### IMPORT ONNX MODEL  
model_path = 'resnet152-v2-7.onnx'
model = onnx.load(model_path)
#########################################################
#########################################################

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# onnxruntime.InferenceSession(path/to/model, providers=['CUDAExecutionProvider']).
session = ort.InferenceSession(model.SerializeToString())

#########################################################
#########################################################
## READ IMAGEN
def get_image(path, show=False):
    with Image.open(path) as img:
        img = np.array(img.convert('RGB'))
    if show:
        plt.imshow(img)
        plt.axis('off')
    return img
#########################################################
#########################################################
## PREPROCESS IMAGE
def preprocess(img):
    img = img / 255.
    img = cv2.resize(img, (256, 256))
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img
#########################################################
#########################################################
## PREDICTS
def predict(path):
    img = get_image(path, show=True)
    img = preprocess(img)
    ort_inputs = {session.get_inputs()[0].name: img}
    preds = session.run(None, ort_inputs)[0]
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    vectors.append(array.array("f",preds))

connection = oracledb.connect(
    user="testvector",
    password="oracle",
    dsn="localhost/freepdb1"
)

vectors = []

directory='/home/oracle/onnx/muestreo/'
pos=0
contenido = os.listdir(directory)

for fichero in contenido:
    if os.path.isfile(os.path.join(directory, fichero)) and fichero.endswith('.jpg'):
        print('...'+fichero)
        predict('muestreo/'+fichero)
        rows = [ (pos, fichero, directory , vectors[pos]) ]
        with connection.cursor() as cursor:
            cursor.executemany("insert into  t_Muestreo_Imagenes (id,nombre_muestra,desc_muestra, vector_muestra) values (:1, :2, :3, :4)",
            rows,
            )
        connection.commit()
        pos += 1
        print('fin...'+fichero)
        

    