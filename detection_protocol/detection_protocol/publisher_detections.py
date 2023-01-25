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

from detection_interfaces.msg import Detections     
from detection_interfaces.msg import Detected 

def detection_to_msg(type, x, y, z):
    detection_msg = Detected()
    detection_msg.type = type
    detection_msg.position.x = x
    detection_msg.position.y = y
    detection_msg.position.z = z       
    return detection_msg 

def msg_to_detection(detection_msg):
    type = detection_msg.type
    x = detection_msg.position.x
    y = detection_msg.position.y 
    z = detection_msg.position.z       
    return type, x, y, z 



def print_detections(detections_msg):
    size = range(len(detections_msg.detected))

    for i in size:
        type, x, y, z = msg_to_detection(detections_msg.detected[i])
        print("-Type: ", type)
        print("Position:")
        print("     x: ", x)
        print("     x: ", y)
        print("     x: ", z)


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('publisher_detections')
        self.publisher_ = self.create_publisher(Detections, 'detections', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Detections()
        detections = range(5)
        msg.header.stamp = self.get_clock().now().to_msg()

        list_det = ["weed", "crop"]
        
        
        for i in detections:
            msg.detected.append(detection_to_msg(random.choice(list_det),
                                        random.uniform(0, 2),
                                        random.uniform(0, 2),
                                        random.uniform(0, 2)))

        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: ')
        timestamp = Time.from_msg(msg.header.stamp)
    
        print("Timestamp in nanoseconds: ", timestamp.nanoseconds)
        print_detections(msg)


        


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
