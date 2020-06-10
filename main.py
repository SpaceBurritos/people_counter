"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
persons_frames= [[69, 198], [228, 444], [500, 700], [750, 870], [921, 1195], [1240, 1358]]
total_people_frames = 1057

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.", )
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    person_in_frame = False
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            person_in_frame = True
    return frame, person_in_frame

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    
    
    
    ### TODO: Handle the input stream ###
    single_image_mode = False
    
    if args.input == 'CAM':
        input_validated = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_validated = args.input

    # Checks for video file
    else:
        input_validated = args.input
        assert os.path.isfile(args.input), "file doesn't exist"
    
    cap = cv2.VideoCapture(input_validated)
    cap.open(input_validated)
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    
    request = 0
    net_input_shape = infer_network.get_input_shape()
    
    
    total_count = 0
    current_count = 0
    frame_tol = 15
    temp_duration = 0
    counter = 0
    positives = 0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        counter += 1
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        start_infer_time = time.time()
        infer_network.exec_net(request, p_frame)
       
        ### TODO: Wait for the result ###
        if infer_network.wait(request) == 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(request)
            output, person_in_frame = draw_boxes(frame, result, prob_threshold, width, height)
            end_infer_time = time.time()
            if person_in_frame:
                for person in persons_frames:
                    if person[0] < counter < person[1]:
                        positives += 1
                        break
                if current_count == 0:
                    start_time = time.time()
                    total_count += 1
                    client.publish("person", json.dumps({"total": total_count}))
                current_count = 1
                frame_tol = 0
                final_time = time.time()
            else:
                if frame_tol < 15:
                    frame_tol += 1
                    current_count = 1
                    final_time = time.time()
                else:
                    if current_count != 0:
                        client.publish("person/duration", json.dumps({"duration": time.time() - start_time}))
                    current_count = 0
                    start_time = 0
                    final_time = 0
                    temp_duration = 0
                    
            temp_duration = final_time - start_time       
            #current_count, frame_tol, duration, total_count = count_people(person_in_frame, frame_tol, duration, total_count, current_count)
            ### TODO: Extract any desired stats from the results ###
            cv2.putText(output, str(positives/total_people_frames*100)+"%", (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), True)
            cv2.putText(output, str(time.time() - start_time), (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), True)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish("person", json.dumps({"count": current_count}))
            #client.publish("person/duration", json.dumps({"duration": avg_duration}))
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(output)
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_img.jpg', output)

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)
    

if __name__ == '__main__':
    main()
