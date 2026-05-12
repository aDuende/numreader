import os, numpy as np, cv2
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model('app/model/thai_digit_model.h5')
CLASS_MAP = {0:'thai_0',1:'thai_1',2:'thai_2',3:'thai_3',4:'thai_4',5:'thai_5'}

def predict_tile(img_64x64):
    arr = img_64x64.astype(np.float32) / 255.0
    arr = arr.reshape(1, 64, 64, 1)
    pred = model.predict(arr, verbose=0)
    cls = int(np.argmax(pred))
    conf = float(np.max(pred))
    return CLASS_MAP[cls], conf, [round(float(x), 3) for x in pred[0]]

print('=== Training image predictions ===')
for cls in ['thai_0','thai_1','thai_2','thai_3','thai_4','thai_5']:
    base = os.path.join('dataset', cls)
    f = sorted(os.listdir(base))[0]
    img = np.array(Image.open(os.path.join(base, f)).convert('L').resize((64, 64)))
    label, conf, probs = predict_tile(img)
    print(f'  {cls}: predicted={label} conf={conf:.2f}  probs={probs}')

print()
print('=== Debug tile predictions ===')
for i in range(2):
    path = f'debug_images/tile_{i}.png'
    if os.path.exists(path):
        img = np.array(Image.open(path).convert('L'))
        label, conf, probs = predict_tile(img)
        print(f'  tile_{i}: predicted={label} conf={conf:.2f}  probs={probs}')
        print(f'    stats: min={img.min()} max={img.max()} mean={round(float(img.mean()),1)} shape={img.shape}')

        # Also save a version scaled up so we can see it better
        big = Image.fromarray(img).resize((256, 256), Image.NEAREST)
        big.save(f'debug_images/tile_{i}_big.png')
        print(f'    Saved debug_images/tile_{i}_big.png')
