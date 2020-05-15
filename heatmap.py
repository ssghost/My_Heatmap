import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import cv2
import time

class Heatmap:
    def __init__(self):
        self.model = None
        self.image = None
        self.size = None
        self.layers = None
        self.lcl = ''
        self.cfl = []
        self.result = None
    
    def image_array(self, imagepath):
        img = keras.preprocessing.image.load_img(imagepath, target_size=self.size)
        array = keras.preprocessing.image.img_to_array(img)
        self.image = np.expand_dims(array, axis=0)

    def read_model(self, modelpath):
        self.model = keras.models.load_model(modelpath)
        self.size = self.model.input_shape
        self.layers = [[layer.name, layer.type] for layer in self.model.layers]
        i = len(self.layers)-1
        classifiers = []
        while self.layers[i][1] != 'Conv2D':
            classifiers.append(self.layers[i][0])
            i-=1
        self.cfl = classifiers
        self.lcl = self.layers[i][0]

    def create_heatmap(self):
        lclayer = self.model.get_layer(self.lcl)
        lclayer_model = keras.Model(self.model.inputs, lclayer.output)
        cfl_input = keras.Input(shape=lclayer.output.shape[1:])
        X = cfl_input
        for name in self.cfl:
            X = self.model.get_layer(name)(X)
        cflayer_model = keras.Model(cfl_input, X)
        with tf.GradientTape() as tape:
            lcl_output = lclayer_model(self.image)
            tape.watch(lcl_output)
            predict = cflayer_model(lcl_output)
            p_index = tf.argmax(predict[0])
            c_channel = predict[:, p_index]
        gradients = tape.gradient(c_channel, lcl_output)
        g_pool = tf.reduce_mean(gradients, axis=(0,1,2))
        lcl_output = lcl_output.numpy()[0]
        g_pool = g_pool.numpy()
        for i in range(g_pool.shape[-1]):
            lcl_output[:,:,i] *= g_pool[i]
        result = np.mean(lcl_output, axis=-1)
        result = np.where(result>=0,result,0)/np.max(result)
        self.result = result
        
    def display_heatmap(self, imagepath, outpath):
        predict = self.model.predict(self.image)
        plt.matshow(self.result)
        plt.title(str(predict))
        plt.show()
        time.sleep(.5)
        img = cv2.imread(imagepath)
        heatmap = cv2.resize(self.result, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        save_img = heatmap*0.4+img
        cv2.imwrite(outpath, save_img)
        time.sleep(.5)
        out = cv2.imread(outpath)
        plt.imshow(out)
        plt.axis('off')
        plt.show() 




                
        
        
