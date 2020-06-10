# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

As mentioned in the Intel(r) Edge AI for IoT Developers Nanodegree Program from Udacity: "Custom layers are those outside of the list of known, supported layers, and are typically a rare exception. Handling custom layers in a neural network for use with the Model Optimizer depends somewhat on the framework used; other than adding the custom layer as an extension, you otherwise have to follow instructions specific to the framework."
Some of the specific instructions that need to be follow are:
1. For Tensorflow, replace the unsupported subgraph with a different subgraph.
2. For Caffe, register the layers as Custom layers, and use Caffe to calculate the output shape


## Comparing Model Performance

The model that I chose was the SSD Mobilenet v2, that i downloaded from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md, the link to the model is http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz. 
After downloading the model use the next command:
```
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
```

And the model was converted to the IR representation using the next command line:
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```
And to run the model I used the following command:
```
python3.5 main.py -d MYRIAD -i resources/Pedestrian_Detect_2_1_1.mp4 -m your-model.xml -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

My method(s) to compare models before and after conversion to Intermediate Representations
were to see the difference in the models accuracy, size, and inference time. For the case of the accuracy
I got the number of frames where there was a person on the screen and compared it to the number of frames
the model infered that there was a person

The difference between model accuracy pre- and post-conversion was 95% and 92%, that is a 3% difference in accuracy

The size of the model pre- and post-conversion was 66.4 MB before and 64.2 MB after, which is a 2.2 MB difference

The inference time of the model pre- and post-conversion was 0.8 s before and 0.73 s after, which is a 0.07 s difference

## Assess Model Use Cases

Some of the potential use cases of the people counter app are knowing the amount of people that are inside a building, or even crossing a bridge or riding an elevator this forsafety measurements; checking the amount of people in a place and a given time, this for the purpose of knowing the growth of a place and its most popular hours. Also, this could be use in places like roller coasters for knowing if the minimum amount of people are in the attraction, and if combined with a face recognition model it can also be used in theaters and movie theaters, to know attendance ratio and to check if the seats are being used by their respective buyers


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. All these variations affect the final model accuracy, because they distort the way that the target (in this case, people) looks and in general makes the whole image less clear.

