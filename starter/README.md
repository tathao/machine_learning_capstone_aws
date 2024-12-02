# Inventory Monitoring at Distribution Centers

Distribution centers are very important in the supply chain. They support the storage, sorting and dispatching the
products. The efficient operations of these centers are paramount to ensure timely deliveries, minimize costs and
maintain inventory accuracy. Before, inventory management based on manual process, which are prone to errors
and inefficiencies. Now, with automation and robotics, distribution centers can adopt robotic system to handle tasks
such as moving objects or managing bins etc..., and robots can automatically handle repetitive task than humans

The primary problem is we don’t have an automated, reliable system for counting the objects within each bin
managed by robots in distribution centers. This project aim to solve that problem. We will focus on develop a
machine learning model that can accurately classify the objects based on images data using the provided Amazon
Bin Image Dataset.

## Project Set Up and Installation
This project contains the following files and directories:

* **sagemaker.ipynb**: This is the main Jupyter Notebook instance that performs the end-to-end process for obtaining a model.
* **src/*.py**: Here are the files used by the `sagemaker.ipynb` notebook for train, evaluate...
* **ProfilerReport**: Contains the profiler report generated after training.

## Dataset

### Overview
The primary dataset for this project is the **Amazon Bin Image Dataset** which contains half a million bin’s images
containing objects. Each image is paired with metadata file detailing the number of objects, their dimensions and
types

Images show bins with varying numbers and types of objects, enhancing the model’s ability to
generalize across different scenarios

![Image from DS](img/00004.jpg)

### Access
The notebook downloads every file from the list `file_list.json` locally and splits them in train, test, and validation datasets before uploading them into a bucket in S3 such as:
![Access](img/00001.jpg)
![Access](img/00002.jpg)

## Model Training
**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of hyperparameters that you specified and why you chose them. Also remember to evaluate the performance of your model.

## Machine Learning Pipeline
**TODO:** Explain your project pipeline.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
