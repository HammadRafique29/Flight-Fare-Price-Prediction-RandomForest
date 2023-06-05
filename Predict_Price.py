from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pandas as pd
import regex as re


class Categorical_to_Numerical_Converstion():
    def __init__(self, test_dataset, train_dataset_style, dataset_type):
        self.dataset = test_dataset
        self.train_dataset_style = train_dataset_style
        self.dataset_type = dataset_type
    
    #################################################
    # converting FLight Time To Numeric (New Colums)
    #################################################
    
    def Flight_Time_to_numeric(self):
        dep_Hours = []
        dep_Min = []
        dep_timePeriod = []

        des_Hours = []
        des_Min = []
        des_timePeriod = []

        for row in self.dataset['Flight_Time']:
            time = row.split('-')
            depTime = time[0]
            if "am" in depTime:
                dep_timePeriod.append(0)
            else:
                dep_timePeriod.append(1)
            depTime = depTime.replace('am', '').replace('pm', '')
            depTime = depTime.split(':')
            dep_Hours.append(int(depTime[0]))
            dep_Min.append(int(depTime[1]))

            desTime = time[1]
            if "am" in desTime:
                des_timePeriod.append(0)
            else:
                des_timePeriod.append(1)
            
            desTime = desTime.replace('am', '').replace('pm', '')
            desTime = desTime.split(':')
            des_Hours.append(int(desTime[0]))
            des_Min.append(int(desTime[1]))

        self.dataset["Depareture_Hour"] = dep_Hours
        self.dataset["Depareture_Min"] = dep_Min
        self.dataset["Depareture_Time_Period"] = dep_timePeriod

        self.dataset["Landing_Hour"] = des_Hours
        self.dataset["Landing_Min"] = des_Min
        self.dataset["Landing_Time_Period"] = des_timePeriod
        return self.dataset
    
    ################################################################
    # converting Date into numeric value (New Columns)
    ###############################################################
    
    def Date_to_numeric(self):
        dep_Year = []
        dep_Month = []
        dep_Day = []

        ret_Year = []
        ret_Month = []
        ret_Day = []

        self.dataset["Departure Date"] = pd.to_datetime(self.dataset['Departure Date'])
        self.dataset["Return Date"] = pd.to_datetime(self.dataset['Return Date'])

        cols = ['Departure Date', 'Return Date']
        for row in self.dataset['Departure Date']:
            Date = str(row).replace(' 00:00:00','').split('-')
            dep_Year.append(Date[0])
            dep_Month.append(Date[1])
            dep_Day.append(Date[2])

        for row in self.dataset['Return Date']:
            Date = str(row).replace(' 00:00:00','').split('-')
            ret_Year.append(Date[0])
            ret_Month.append(Date[1])
            ret_Day.append(Date[2])

        self.dataset["Depareture_Year"] = dep_Year
        self.dataset["Depareture_Month"] = dep_Month
        self.dataset["Depareture_Day"] = dep_Day

        self.dataset["Return_Year"] = ret_Year
        self.dataset["Return_Month"] = ret_Month
        self.dataset["Return_Day"] = ret_Day
        return self.dataset
    
    ################################################################
    # converting Duration into numeric value (New Columns)
    ###############################################################
    
    def Duration_to_numeric(self):
        duration_Hours = []
        duration_Min = []
        for row in self.dataset['Duration']:
            temp_text = re.findall(" \([1-9].[a-z\(\)]+", row)
            row = row.replace(temp_text[0], '').split(' ')
            for data in row:
                if "h" in data:
                    duration_Hours.append(int(data.replace('h', '')))
                    duration_Min.append(int(row[1].replace('m', '')))
                    break
                elif "m" in data:
                    duration_Hours.append(0)
                    duration_Min.append(int(data.replace('m', '')))
                    break         
        self.dataset["Duration_Hour"] = duration_Hours
        self.dataset["Duration_Min"] = duration_Min
        return self.dataset
    
    ################################################
    # Changing Stops Column into Numeric Values
    ################################################

    def Stops_to_numeric(self):
        total_Stops = []
        for data in self.dataset["Stops"]:
            data = data.split('-')
            total_Stops.append(len(data))

        self.dataset["Stops"] = total_Stops
        return self.dataset
    
    ################################################
    # Changing columns datatyes of int64
    ################################################
    
    def change_dtypes(self):
        types = {'Depareture_Hour':'int64',
                       'Depareture_Min':'int64',
                       'Depareture_Time_Period':'int64',
                       'Landing_Hour':'int64',
                       'Landing_Min':'int64',
                       'Landing_Time_Period':'int64',
                        'Depareture_Year':'int64',
                       'Depareture_Month':'int64',
                        'Depareture_Day':'int64',
                        'Return_Year':'int64',
                       'Return_Month':'int64',
                        'Return_Day':'int64',
                       'Duration_Hour':'int64',
                        'Duration_Min':'int64',
                        'Stops': 'int64',
                       }
        self.dataset = self.dataset.astype(types)
        return self.dataset
    

    ################################################
    # Drop Categorical Columns After Converstion
    ################################################

    def drop_dumy_cols(self):
        dropCol = ['Flight_Time','Departure Date', 'Return Date', 'Duration']
        self.dataset.drop(dropCol, inplace=True, axis=1)
        return self.dataset
    
    
    def Conversion_Implementation(self):
        self.Flight_Time_to_numeric()
        self.Date_to_numeric()
        self.Duration_to_numeric()
        self.Stops_to_numeric()
        self.change_dtypes()
        self.drop_dumy_cols()
        return self.dataset

class get_dunnnies_col():
    def __init__(self, dataset):
        self.dataset = dataset
        self.airline = None
        self.Source = None
        self.Destination = None
    
    def flight_Name_Dummies(self):
        cols = ['Etihad Airways', 'Multiple airlines', 'Multiple airlines ', 'Turkish Airlines', 'Qatar Airways', 'Saudi Arabian Airlines', 'Kuwait Airways', 'Emirates', 'Qatar Airways ', 'Flight + Bus', 'British Airways ', 'Pakistan International Airlines ', 'Gulf Air', 'flydubai', 'Gulf Air ', 'Emirates ', 'American Airlines ', 'Turkish Airlines', 'Pakistan International Airlines', 'Etihad Airways ', 'flynas', 'KLM ', 'JetBlue Airways ']
        for name in cols:
            if name == self.dataset['Flight_Name'][0]:
                self.dataset[name][0] = 1
            elif name != self.dataset['Flight_Name'][0]:
                self.dataset[name] = 0

    def flight_Name_Dummies2(self):
        self.Airline = self.dataset['Flight_Name']
        self.Airline = pd.get_dummies(self.Airline, drop_first=True)
        self.Airline = self.Airline.astype(int)
    
    def Origin_Dummies(self):
        self.Source = self.dataset['Origin']
        self.Source = pd.get_dummies(self.Source, drop_first=True)
        self.Source = self.Source.astype(int)
    
    def Destination_Dummies(self):
        self.Destination = self.dataset['Destination']
        self.Destination = pd.get_dummies(self.Destination, drop_first=True)
        self.Destination = self.Destination.astype(int)
    
    def all_dummies_concatenation(self):
        self.dataset = pd.concat([self.dataset, self.Airline, self.Source, self.Destination], axis=1)
        
    def droping_org_columns_after_dummies(self):
        col = ['Flight_Name', 'Origin', 'Destination']
        self.dataset.drop(col, axis=1, inplace=True)
        
    def dummies_implementation(self):
        self.flight_Name_Dummies()
        self.Origin_Dummies()
        self.Destination_Dummies()
        self.all_dummies_concatenation()
        self.droping_org_columns_after_dummies()
        return self.dataset

class values_formating():
    def __init__(self, test_dataset, train_dataset_format):
        self.test_dataset = test_dataset
        self.train_dataset_format = train_dataset_format

    def droping_org_columns_after_dummies(self):
        col = ['Flight_Name', 'Origin', 'Destination']
        self.train_dataset_format.drop(col, axis=1, inplace=True)

    def Implementation(self):
        format_cols = self.train_dataset_format.columns
        test_cols = self.test_dataset.columns
        for row in format_cols:
            if row in test_cols:
                pre_val = self.test_dataset[row][0]
                self.train_dataset_format.loc[0, row] = pre_val
            else:
                self.train_dataset_format.loc[0, row] = 0
            
            if self.test_dataset['Flight_Name'][0] == row:
                self.train_dataset_format.loc[0, row] = 1

            if self.test_dataset['Destination'][0] == row:
                self.train_dataset_format.loc[0, row] = 1
        
        self.test_dataset = self.train_dataset_format.drop('Price', axis=1)
        return self.test_dataset

    
class Prediction():
    def __init__(self, dataset):
        self.dataset = dataset
    
    def predict(self):
        import joblib
        model = joblib.load('Supported_Data/Flight-Fare-Prediction.pkl')
        return model.predict(self.dataset.head(2))




##########################################################################
# Reading Dataset Format Style Sheet For Prediction
##########################################################################


train_dataset_format = pd.read_csv(f"Supported_Data/Dataset_Format.csv")
test_dataset = None

pre_val_method = sys.argv[1]

# Ask the user if he want to import csv file or input data manually

# print("#"*70 + "\nPREDICT VALUE USING CSV FILE OR MANUAL DATA USING INPUT? ")
# dec = input("Type (0 = Dataset File, 1 = Manual Data: ")
# print("#"*70)

# if user choice 1, input data manually using terminal
if pre_val_method == "-m":
    print("#"*70 + "\nPREDICT VALUE USING MANUALL INPUT DATA! \n" + "#"*70)
    input_structure = {'Origin': [], 'Destination': [], 'Flight_Time': [], 'Departure Date': [], 'Return Date': [], 'Duration': [], 'Stops': [], 'Flight_Name': [], 'Price': []}
    for col_name in input_structure.keys():
        input_structure[col_name].append(input(f"Enter {col_name}: "))
        if col_name == "Price":
            input_structure['Price'] = 0
    print("-"*30 + "\n")
    test_dataset = pd.DataFrame(input_structure)

# if user choice 0, than script automatically insert the file that need to 
# cleaned first using Cleaning_Dataset Script.
elif pre_val_method == "-a":
    dataset_name = sys.argv[2]
    test_dataset = pd.read_csv(f"{dataset_name}.csv")


# Now we have our Predicted Dataset for Specific flight,
# We now predict values after performing features coversion to numerical.

conversion_obj = Categorical_to_Numerical_Converstion(test_dataset, train_dataset_format, dataset_type="Test")
dataset = conversion_obj.Conversion_Implementation()

formating = values_formating(dataset, train_dataset_format)
dataset = formating.Implementation()

Prediction = Prediction(dataset)
value = Prediction.predict()
print("The Price: ", int(value) - 66)




