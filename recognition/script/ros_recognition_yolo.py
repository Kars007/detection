import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import cv2

import torch
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

import ros2_numpy as rnp

from detection_interfaces.msg import Detections     
from detection_interfaces.msg import Detected 

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
led_strip = GPIO.PWM(32,500)
led_strip.start(0)

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.pcd_as_numpy_array = np.array([], [])
        self.pcd_subscription = self.create_subscription(PointCloud2, 'zed2/zed_node/point_cloud/cloud_registered', self.listener_callback_pcb, 10)
        self.subscription = self.create_subscription(Image, 'zed2/zed_node/left/image_rect_color', self.listener_callback, 10)
        self.subscription # prevent unused variable warning
        self.pcd_subscription
        self.publisher = self.create_publisher(Image, 'Yolo_result', 10)
        self.publisher_detections = self.create_publisher(Detections, "detections", 10)

        self.bridge = CvBridge()

        device = 'cuda:0'
        set_logging()
        device = select_device('0')
        self.device = device
        half = device.type != 'cpu'
        self.half = half
        weights = 'best.pt'
        imgsz = 416
        led_strip.ChangeDutyCycle(50)
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)
        model = TracedModel(model, device, img_size=imgsz)
        model.half()
        names = model.module.names if hasattr(model, 'module') else model.names
        self.names = names
        print(names)
        colors = [[random.randint(0,255) for _ in range(3)] for _ in names]
        self.colors = colors
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        self.model = model

    def listener_callback(self, data):
        msg = Detections()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info('Receiving image')
        img0 = self.bridge.imgmsg_to_cv2(data)
        img1= cv2.cvtColor(img0, cv2.COLOR_RGBA2RGB) #change rgba image to rgb, needed for the yolo model to work
        img = img1[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
        conf_thres = 0.1
        iou_thres = 0.45
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        for i, det in enumerate(pred):  # detections per image
            s = ""
            gn = torch.tensor(img0.shape)[[1,0,1,0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    i, j = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))  #get the pixels of the bounding box
                    center_point = round((i[0]+j[0])/2), j[1]    #get the middle point od the bounding box
                    circle = cv2.circle(img0, center_point, 1, (0,0,255), 1)  #place a circle at the middle point
                    xyz_pos = self.detection_average(int(center_point[0]), int(center_point[1]))
                    if xyz_pos[0] == 0 or xyz_pos[0] == np.nan:
                         continue    # When a xyz position has 0 it inclinse that the average that was calculate was done with only NaN, which is why this detection will be disregarded
                    label = f'{self.names[int(cls)]} {conf:.2f} {xyz_pos}'
                    plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=1)
                    msg.detected.append(detection_to_msg(self.names[int(cls)], xyz_pos[0], xyz_pos[1], xyz_pos[2]))
                self.publisher_detections.publish(msg)
                self.publisher.publish(self.bridge.cv2_to_imgmsg(img0, "bgra8"))

    def listener_callback_pcb(self, data):
        self.pcd_as_numpy_array = rnp.point_cloud2.get_xyz_points(rnp.point_cloud2.pointcloud2_to_array(data), remove_nans=False)

    def detection_average(self, i, j):
        average_array= np.zeros(shape=(9,3))
        n = 0
        for x in range(-1,2):
            if j > 510:  #if point outsite of picture it will fill that part of the array with NaN
                average_array[n] = [np.nan, np.nan, np.nan]
                average_array[n+1] = [np.nan, np.nan, np.nan]
                average_array[n+2] = [np.nan, np.nan, np.nan]
                n+=3
                continue
            average_array[n] = [self.pcd_as_numpy_array[(j+x)][(i-1)][0], self.pcd_as_numpy_array[(j+x)][(i-1)][1], self.pcd_as_numpy_array[(j+x)][(i-1)][2]]
            average_array[n+1] = [self.pcd_as_numpy_array[(j+x)][i][0], self.pcd_as_numpy_array[(j+x)][i][1], self.pcd_as_numpy_array[(j+x)][i][2]]
            average_array[n+2] = [self.pcd_as_numpy_array[(j+x)][(i+1)][0], self.pcd_as_numpy_array[(j+x)][(i+1)][1], self.pcd_as_numpy_array[(j+x)][(i+1)][2]]
            n+=3
        return np.nanmean(average_array, axis=0)

def detection_to_msg(type, x, y, z):
    detection_msg = Detected()
    detection_msg.type = type
    detection_msg.position.x = x
    detection_msg.position.y = y
    detection_msg.position.z = z       
    return detection_msg 

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    led_strip.ChangeDutyCycle(99)
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
