import colorsys, os, cv2
import pyrealsense2 as rs
import numpy as np
import math

from PIL import Image
from PIL import ImageFont, ImageDraw

import roslib
import rospy
from sensor_msgs.msg import Image as Im
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion, TransformStamped, PoseArray

from yolo import YOLO

class objectRecognition():
    def __init__(self):
        rospy.init_node("ObjectRecognition", anonymous = True)
        self.color_img = None
        self.depth_img = None
        self.yolo = YOLO()
        self.bridge = CvBridge()
        self.keystroke = 0

        self.pos_pub = rospy.Publisher('obj_pos',Point, queue_size=2)
        # for gazebo simulation
        rospy.Subscriber('/realsense/color/image_raw',Im, self.image_callback)
        rospy.Subscriber('/realsense/depth/image_rect_raw',Im, self.depth_callback)
        
        # for realsense camera
        #rospy.Subscriber('/camera/color/image_raw',Im, self.image_callback)
        #rospy.Subscriber('/camera/depth/image_rect_raw',Im, self.depth_callback)

        # for astra camera
        #rospy.Subscriber('/camera/rgb/image_raw',Im, self.image_callback)
        #rospy.Subscriber('/camera/depth/image_raw',Im, self.depth_callback)

    def image_callback(self,data):
        try: 
            self.color_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #print("Color ",np.shape(self.color_img))
        except CvBridgeError:
            print("CvBridge could not convert images from Camera to opencv")
        
    def depth_callback(self, data):
        try:
            # The depth image is a single-channel float32 image
            self.depth_img = self.bridge.imgmsg_to_cv2(data, "32FC1")
            #print("Depth ",np.shape(self.depth_img))
        except CvBridgeError as e:
            print(e)

    def detect(self):
        if((self.color_img is not None) and (self.depth_img is not None)):
            depth_array = np.array(self.depth_img, dtype=np.float64)
            color_array = np.array(self.color_img, dtype=np.uint8)
            color_array2= Image.fromarray(self.color_img)   

            img, obj_pose_dict = self.yolo.detect_image(color_array2, depth_array)
            #print(obj_pose_dict)
            try:
                self.pose = obj_pose_dict['fire hydrant']
                print("Found")
            except:
                self.pose = (500, 320, 1)

            if(math.isnan(self.pose[2])):
                self.pose[2] = 10
                
            self.obj_pose = Point()
            self.obj_pose.x = self.pose[0]
            self.obj_pose.y = self.pose[1]
            self.obj_pose.z = self.pose[2]
            
            print(self.pose)
            result = np.asarray(img)

            cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)

            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            
            cv2.imshow("Color Image", color_array)
            cv2.imshow("Depth Image", depth_array)
            cv2.imshow("Result", result)

            self.keystroke = cv2.waitKey(1)
            #print()
            if 32 <= self.keystroke and self.keystroke < 128:
                cc = chr(self.keystroke).lower()
                if cc == 'q':
                    # The user has press the q key, so exit
                    rospy.signal_shutdown("User hit q key to quit.")

    def spin(self):
        rospy.loginfo("\n\n***Initiating Object Recognition Node***\n\n")
        rate = rospy.Rate(2)
        rospy.on_shutdown(self.shutdown)

        self.pose = (500, 320, 1)
        self.obj_pose = Point()
        self.obj_pose.x = 320
        self.obj_pose.y = 320
        self.obj_pose.z = 1

        while not rospy.is_shutdown():
            self.detect()
            self.pos_pub.publish(self.obj_pose)
            rate.sleep()
        rospy.spin()
    
    def shutdown(self):
        rospy.loginfo("\n\n***Terminating Object Recognition Node***\n\n")
        cv2.destroyAllWindows()
        self.yolo.close_session()

def main():
    obj = objectRecognition()
    try:
        obj.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__=='__main__':
    main()
