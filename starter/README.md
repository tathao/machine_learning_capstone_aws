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
![Access](img/00001.png)
![Upload](img/00002.png)

## Model Training
In this project, the **EfficientNet** model was chosen because its superior performance. EfficientNet uses a compound scaling method to balance width, depth, and resolution, which allows it to outperform many traditional architectures like ResNet on image classification tasks while being computationally lighter. The Efficient accuracy is better with fewer parameters and less computational overhead than Restnet50

Hyperparameters type:

* EfficientNet was initialized with weights pretrained on ImageNet to leverage learned feature representations. This significantly reduces training time and the risk of overfitting, especially on small datasets.
* While fine-tuning, all layers except the final classifier were frozen to retain the pretrained features and avoid overfitting in early layers. This allowed the model to focus on learning dataset-specific features in the classifier.
* A fully connected layer (nn.Linear) maps feature vectors to the number of output classes. Dropout (p=0.5) adds regularization and batch normalization improves stability and convergence. They all set to prevent overfitting and helps the model generalize well.

Evaluate the performance:

* Accuracy: EfficientNet (31%) > ResNet50 (29%)
* Precision: EfficientNet (0.34) > ResNet50 (0.21)
* Recall: EfficientNet (0.30) > ResNet50 (0.27)
* F1 Score: EfficientNet (0.30) > ResNet50 (0.19)

## Machine Learning Pipeline
This project pipline on the following:

* Data download: The dataset is downloaded to local then upload to S3, so other instances can use.
* Data transformation: The images were resized to 224x224 pixels and rotated randomly before training the model
* Model selection: Use a pre-trained EfficientNet as a baseline model. Hyperparameter tuning was performed with 5 jobs and then the best model was selected to be trained.
* Model training: the best model was trained and evaluated
* Debug and Profiling: a complete profiler report was obtained with relevant information such as GPU/CPU utilization and debugger information such as training loss was reported.
* Model deployment: Deploy the best model to a SageMaker endpoint.
* Model Inference: Can be used to make predictions for different images

## Standout Suggestions
**(Optional):** This is where you can provide information about any standout suggestions that you have attempted.

* Hyperparameter Tuning: When selecting hyperparameters and their ranges for tuning, the goal is to focus on those that significantly influence the model's performance while ensuring the search space is broad enough to explore promising configurations efficiently

    * Batch Size: The range [32, 64, 128]
    * Learning Rate: The range [1e-4, 5e-2]
    * Number of Epochs: [5, 15]
* Model Profiling and Debugging: Use model debugging and profiling to better monitor and debug your model training job.

```
#Declare your HP ranges, metrics etc.
hyperparameter_ranges = {
    'batch_size': IntegerParameter(32, 128),
    'learning_rate': ContinuousParameter(1e-4, 5e-2),
    'num_epochs': IntegerParameter(5, 15)
}

objective_metric_name = 'Average loss'
objective_type = 'Minimize'

# Define the metric regexes to capture 'Average loss', 'Precision', 'Recall', and 'F1 Score'
metric_definitions = [
    {"Name": "Average loss", "Regex": "Average loss: ([0-9\\.]+)"},
    {"Name": "Precision", "Regex": "Precision: ([0-9\\.]+)"},
    {"Name": "Recall", "Regex": "Recall: ([0-9\\.]+)"},
    {"Name": "F1 Score", "Regex": "F1 Score: ([0-9\\.]+)"}
]
```
* Model Deploying and Querying
