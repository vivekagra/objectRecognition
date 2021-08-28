import colorsys, os, cv2
# import pyrealsense2 as rs
import numpy as np
# import sys, time
# from PIL import Image, ImageFont, ImageDraw
# from timeit import default_timer as timer

import roslib
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

color_img = None
class data_reader():
    def __init__(self):
        self.bridge = CvBridge()
        # self.sub = rospy.Subscriber('/realsense/color/image_raw',Image, self.callback)
        self.sub = rospy.Subscriber('/camera/color/image_raw',Image, self.callback)
        # self.sub = rospy.Subscriber('')

    def callback(self,data):
        try: 
            color_img = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
            print("CvBridge could not convert images from realsense to opencv")
        
        # height,width, channels = my_image.shape                               f   fg  
        # my_height = my_image.shape[0]
        #print(color_img.shape)
        cv2.imshow("result", color_img)
        cv2.waitKey(1)

def main():
    d = data_reader()
    rospy.init_node("Data_Reader",anonymous = True)
    try:
        
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
    rospy.spin()