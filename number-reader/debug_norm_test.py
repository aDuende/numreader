"""Test the updated _make_tile normalization on the last saved canvas."""
import os, sys, io, numpy as np, cv2
from PIL import Image

sys.path.insert(0, '.')
from app.utils.split_digits import split_digits_from_bytes
import tensorflow as tf

model = tf.keras.models.load_model('app/model/thai_digit_model.h5')
CLASS_MAP = {0:'thai_0(๐)',1:'thai_1(๑)',2:'thai_2(๒)',3:'thai_3(๓)',4:'thai_4(๔)',5:'thai_5(๕)'}

canvas_path = 'debug_images/last_canvas.png'
with open(canvas_path, 'rb') as f:
    tiles = split_digits_from_bytes(f.read())

print(f"Tiles produced: {len(tiles)}")
for i, tile in enumerate(tiles):
    mean = float(tile.mean())
    coverage = round(1.0 - mean / 255.0, 3)
    print(f"  tile_{i}: mean={round(mean,1)} coverage={coverage}")
    arr = tile.astype(np.float32) / 255.0
    pred = model.predict(arr.reshape(1,64,64,1), verbose=0)
    cls = int(np.argmax(pred))
    conf = float(np.max(pred))
    print(f"    predicted={CLASS_MAP[cls]} conf={conf:.2f}")
    Image.fromarray(tile).resize((256,256), Image.NEAREST).save(f'debug_images/tile_{i}_norm.png')
    print(f"    saved debug_images/tile_{i}_norm.png")
