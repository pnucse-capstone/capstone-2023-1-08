
import cv2
from PIL import Image
import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from glob import glob
import time
import cv2
from models.cyclegan.cyclegan_model import CycleGANModel, get_transform, tensor2im
from models.cyclegan.whole_test_option import WholeOption
import torch


class mirnet:
    def __init__(self, size):
        self.model = tf.keras.models.load_model("models/mirnet", compile=False)
        self.desired_size = size
    def infer(self, image):
        preprocess_image ,original_image = self.preprocessing(image)
        curr= time.time()
        enhanced_image =  self.model.predict(preprocess_image)
        print(f"image inference time : {time.time()-curr}")
        return self.postprocessing(original_image, enhanced_image)

    def preprocessing(self, image):
        curr = time.time()
        original_image = cv2.resize(image, dsize=self.desired_size, interpolation=cv2.INTER_LINEAR)
        img = keras.utils.img_to_array(original_image)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)    
        print(f"image preprocessing time : {time.time()-curr}")
        return img, original_image
    
    def postprocessing(self, image_before, image_resize):
        curr= time.time()
        output = image_resize[0].reshape(
        (self.desired_size[0], self.desired_size[1], 3)
    )
        output_image = output * 255.0
        output_image = output_image.clip(0, 255)
        image_after = Image.fromarray(np.uint8(output_image))
        image_after = np.uint8(image_after)
        print(f"image postprocessing time : {time.time()-curr}")
        return image_after
        
class cycle:
    def __init__(self, size):
        self.opt = WholeOption().parse()  # get test options
        self.model = CycleGANModel(self.opt)
        self.model.load_networks()
        self.desired_size = size
        if self.opt.eval:
          self.model.eval()

    def infer(self, image):
        original_image = self.preprocessing(image)
        curr= time.time()
        original_array = Image.fromarray(original_image)
        transform = get_transform(self.opt)
        original_array = transform(original_array)
        original_array = torch.unsqueeze(original_array, 0)

        self.model.set_input(original_array)
        with torch.no_grad():
          self.model.forward()
        enhanced_image =  self.model.fake_B
        print(f"image inference time : {time.time()-curr}")
        return self.postprocessing(original_image, enhanced_image)

    def preprocessing(self, image):
        curr = time.time()
        image = cv2.resize(image, dsize=self.desired_size, interpolation=cv2.INTER_LINEAR)
        print(f"image preprocessing time : {time.time()-curr}")
        return image
    
    def postprocessing(self, image_before, image_after):
        curr= time.time()
        rst_img = tensor2im(image_after)
        print(f"image postprocessing time : {time.time()-curr}")
        return rst_img