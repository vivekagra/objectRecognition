# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""


from timeit import default_timer as timer

import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from PIL import Image
from PIL import ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from tensorflow.keras.utils import multi_gpu_model

import colorsys, os, cv2
import pyrealsense2 as rs
import numpy as np
import sys, time
from scipy import stats

import roslib
import rospy
from sensor_msgs.msg import Image as Im
from cv_bridge import CvBridge, CvBridgeError

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        
        self.model_path = 'model_data/yolo.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.3
        self.iou =  0.45
        self.model_image_size = (416, 416)
        self.gpu_num = 1
        
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        with self.sess.as_default():
            with self.sess.graph.as_default():
                self.K_learning_phase = K.learning_phase()


        
        self.boxes, self.scores, self.classes = self.generate()



    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def pixelToPose(self, depth_img, top, left, bottom, right):
        w = 200
        h = 200
        hfov = 69.4
        vfov = 42.5
        pi = 3.14
        f_x = w/(2*np.tan(hfov*pi/360))
        f_y = h/(2*np.tan(vfov*pi/360))
        
        cx = int((left+right)/2)
        cy = int((top+bottom)/2)
        z = depth_img[cy][cx]
        # n = 3
        # kernel = np.zeros((n*n),dtype=np.float32)
        # for i in range(n):
        #     for j in range(n):
        #         kernel[n*i+j] = depth_img[cy-n+1+i][cx-n+1+j]
        
        # mode = stats.mode(kernel)[0][-1]
        # mean = np.mean(kernel)
        # if(mode>=mean):
        #     z = mode
        # else:
        #     z = mean

        # z = z/1000 # depth in metres
        
        x = (cx - (w/2))*(z/(f_x))
        y = (cy - (h/2))*(z/(f_y))
        
        return (cx,cy,z)


    def detect_image(self, color_img, depth_img):
        start = timer()
        
        #image_data = np.array(boxed_image, dtype='float32')
        image_data = np.array(color_img, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [color_img.size[1], color_img.size[0]],
                self.K_learning_phase: 0
            })

        # print('\n\nFound {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * color_img.size[1] + 0.5).astype('int32'))
        thickness = (color_img.size[0] + color_img.size[1]) // 300
        
        obj_pose_dict = {}

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(color_img)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(color_img.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(color_img.size[0], np.floor(right + 0.5).astype('int32'))
            
            pose = self.pixelToPose(depth_img, top, left, bottom, right)
            obj_pose_dict[predicted_class] = pose
            #print(label, "is present at", pose)

            #print(label, "is present in box", (top, left), (bottom, right), "at a distance of", depth, "metres")

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return color_img, obj_pose_dict

    def close_session(self):
        self.sess.close()    