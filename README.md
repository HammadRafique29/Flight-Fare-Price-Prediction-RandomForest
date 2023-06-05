# Flight-Fare-Price-Prediction-RandomForest
Predicting The Fare Price of Flights Using RandomFores Model. (Artificial Intelligence)

## Requirement:
- sklearn
- numpy
- seaborn
- matplotlib
- sys
- pandas
- regex

## Instruction:

Training Script is written according the specific columns format, you can view the format in org_dataset.csv file.

After you set your dataset file according to format given in org_dataset.csv format
Next Step is to Clean the Dataset using Cleaning_Dataset.py Module, by providing the dataset_name and dataset_type (Training or Testing) as command line argument, dataset_type differentiate between training dataset and testing dataset (used for predicting price).

After you Cleaned You Dataset, Now its to Train your Model, run Train_Model Module and pass dataset_name as command line argument, and wait for a minutes. After completion Model Will be saved at Supported_data folder.

Now it's time to predict the price for specific input values, You need to run Predict_Price Module by passing any commands which is as follow:

- -m  (m means insert data manually using terminal Input)
- -a  (a means automatically, you have input values in dataset, you have pass it using this argument)

### python Predict_price.py -a predict_dataset_name   
(This will import predict_dataset_name file and perform operation automatically and will display the price
