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
    def __init__(self, dataset, dataset_type):
        self.dataset = dataset
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
        if self.dataset_type == "Train":
            types['Price'] = 'int64'
            self.dataset = self.dataset.astype(types)
        elif self.dataset_type == "Test":
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


class Apply_RandomForestModel():
    def __init__(self, train_dataset):
        self.training_dataset = train_dataset
        self.rfr = None
        self.random_search = None
    
    def split_dataset(self):
        self.x = training_dataset.drop('Price', axis=1)
        self.y = training_dataset['Price']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=51)

    def Model_Fitting(self):
        self.rfr = RandomForestRegressor()
        self.rfr.fit(self.x_train, self.y_train)

    def Prediction(self):
        self.prediction = self.rfr.predict(self.x_test)
        print(self.prediction[:5])

    def test_score(self):
        score = self.rfr.score(self.x_test, self.y_test)
        print(f"(X, Y) Test Score: {score}")
    
    def Model_Error(self):
        print('MAE:', metrics.mean_absolute_error(self.y_test, self.prediction))
        print('MSE:', metrics.mean_squared_error(self.y_test, self.prediction))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(self.y_test, self.prediction)))
    
    def R2_score(self):
        score = metrics.r2_score(self.y_test,self.prediction)
        print(f"R2 Score: {score}")

    def plotting_Model(self):
        plt.figure(figsize = (8,8))
        sns.displot(self.y_test - self.prediction)
        plt.show()

        sns.displot(self.y_test - self.prediction)
        plt.scatter(self.y_test , self.prediction , alpha=0.8)
        plt.xlabel('y_test')
        plt.ylabel('pred')

    def training(self):
        print(f"Training Process Started! It might Take some Time .....")
        n_estimators = [int(x) for x in np.linspace(100,2000,10)]
        max_depth = [int(x) for x in np.linspace(100,2000,10)]
        min_samples_split=[2,4,6,8,10,12,14]
        min_samples_leaf=[1,3,5,7,8,10]
        max_features=['sqrt','log2',1.0,None]

        self.random_search = {
            'n_estimators' : n_estimators,
            'max_depth' : max_depth,
            'min_samples_split' : min_samples_split,
            'min_samples_leaf' : min_samples_leaf,
            'max_features' : max_features, 
        }

        print("\t", self.random_search)
        process = self.rfr_random = RandomizedSearchCV(estimator=self.rfr , param_distributions=self.random_search , n_iter=10 , cv=5 , verbose=2 , random_state=51 , n_jobs=1)
        print("\t", process)
        self.rfr_random.fit(self.x_train , self.y_train)

        best_parameters = self.rfr_random.best_params_
        print(f"Best Fit Parameters: {best_parameters}")

    def save_Model(self):
        import pickle
        file = open('Supported_Data/Flight-Fare-Prediction.pkl','wb')
        pickle.dump(self.rfr_random,file)
        print('#'*50 + "\n\tMODEL HAS BEEN SAVED SUCCESSFULLY ......\n" + "#"*50)
    
    def save_dataset(self):
        self.training_dataset.to_csv('Supported_Data/Trained_Dataset.csv', sep=',', index=False)
        return self.training_dataset

    def Implementation(self):
        self.split_dataset()
        self.Model_Fitting()
        self.Prediction()
        self.test_score()
        self.Model_Error()
        self.R2_score()
        self.plotting_Model()
        self.training()
        self.save_Model()
        return self.save_dataset()
    # def random_prediction(self):
    #     predict = rfr_random.predict(x_test)
    #     print(predict)




try:
    dataset_name = sys.argv[1]

    print('#'*50 + "\n\tCSV FILE IMPORTED SUCCESSFULLY ......\n" + "#"*50)
    training_dataset = pd.read_csv(f"Supported_Data/Cleaned_{dataset_name}.csv")

    # ###########################################################################################
    # # Categorical to Numerical / Creating Dummies Columns For Training Dataset
    # ###########################################################################################

    # -------------------------------------
    # Categorical to Numerical Conversion
    # -------------------------------------

    Training_Dataset_Conversion_Obj = Categorical_to_Numerical_Converstion(training_dataset, "Train")
    training_dataset = Training_Dataset_Conversion_Obj.Conversion_Implementation()

    # -------------------------------------
    # Creating Dummies Columns
    # -------------------------------------

    Training_Dataset_Dummies_Obj = get_dunnnies_col(training_dataset)
    training_dataset = Training_Dataset_Dummies_Obj.dummies_implementation()


    RFM = Apply_RandomForestModel(training_dataset)
    training_dataset = RFM.Implementation()
    training_dataset.head()

except Exception as e:
    print(e)