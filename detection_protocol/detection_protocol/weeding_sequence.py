# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
import random
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import TwistStamped
import message_filters
from detection_interfaces.msg import Detections     
from detection_interfaces.msg import Detected 

def detection_to_msg(type, x, y, z):            # Makes a detection ready t be sent, converts variables to a msg Detected() this message can then be placed in in detections message
    detection_msg = Detected()
    detection_msg.type = type
    detection_msg.position.x = x
    detection_msg.position.y = y
    detection_msg.position.z = z       
    return detection_msg 

def msg_to_detection(detection_msg):            # Detection messaged converted back to variables
    type = detection_msg.type
    x = detection_msg.position.x
    y = detection_msg.position.y 
    z = detection_msg.position.z       
    return type, x, y, z 

def print_detections(detections_msg):           # Prints the detections in terminal
    size = range(len(detections_msg.detected))

    for i in size:
        type, x, y, z = msg_to_detection(detections_msg.detected[i])
        print("-Type: ", type)
        print("-Position:")
        print("     x: ", x)
        print("     y: ", y)
        print("     z: ", z)
    
def detections_to_list(detections):             # Detections msg to a list of variables, this function uses the function msg_to_detection
    size = range(len(detections.detected))
    list = []
    for i in size:
        type, x, y, z = msg_to_detection(detections.detected[i])
        object = [type, x, y, z ]
        # object.append(type)
        # object.append(x)
        # object.append(y)
        # object.append(z)
        list.append(object)
    return list

def get_x(detection):                           # This functions takes the x positions of the detection, this function will be used in order_detections_x
    return detection[1]

def order_detections_x(detections):             # Here the detections will be placed in order of lowest x value to highest
    list = detections_to_list(detections)
    list.sort(key=get_x)
    return list

def publish_filter_data(self, OldDetections):   # Here all the functions will be combined to use the incoming data and filter it and then publish it again to the topic
    msg = Detections()
    size = range(len(OldDetections.detected))
    Ordered_detections = order_detections_x(OldDetections)

    for i in size:
        object = Ordered_detections[i]
        msg.detected.append(
            detection_to_msg(
                    object[0],
                    object[1],
                    object[2],
                    object[3]))

    self.publisher.publish(msg)
    self.get_logger().info('Publishing: ')
    timestamp = Time.from_msg(OldDetections.header.stamp)
    
    print("Timestamp in nanoseconds: ", timestamp.nanoseconds)
    print("Speed: %f m/s" % self.speed) 
    print_detections(msg)    

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('weeding_sequence')
        self.speed = 0.0

        self.publisher = self.create_publisher(     #create publisher
            Detections, 
            'weeding_sequence', 
            10)
        
        self.sub_speed = self.create_subscription(  #create subscriber
            TwistStamped,
            'speed',
            self.listener_callback_speed,
            10)
        self.sub_speed # prevent unused variable warning 

        self.sub_detections = self.create_subscription( #create subscriber
            Detections,
            'detections',
            self.listener_callback_detections,
            10)
        self.sub_detections # prevent unused variable warning

    def listener_callback_speed(self, Speed):           # save speed from speed topic
        self.speed =  float(Speed.twist.linear.x)  

    def listener_callback_detections(self, Detections): # obtain detections from detections topic and execute publish_filter_data function
        self.get_logger().info('I heard: ')
        timestamp = Time.from_msg(Detections.header.stamp)
    
        print("Timestamp in nanoseconds: ", timestamp.nanoseconds)
        print("Speed: %f m/s" % self.speed) 
        print_detections(Detections)
        print("-----------------------------------")
        publish_filter_data(self, Detections)

def main(args=None):
    
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
