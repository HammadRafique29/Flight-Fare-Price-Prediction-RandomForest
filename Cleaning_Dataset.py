import pandas as pd

###################################################################
# Fixing Column values in Dataset (FlightName, Date, Stops, Price)
###################################################################

class cleaning():
    def __init__(self, dataset, drop_cols, dataset_type):
        self.dataset = dataset
        self.drop_cols = drop_cols
        self.dataset_type = dataset_type
    
    def drop_col(self):
        self.dataset.drop(self.drop_cols, axis=1, inplace=True)
        
    def fix_Name_col(self):
        flightNames = []
        for row in dataset['Flight_Name']:
            row = row.split('� ')[0]
            flightNames.append(row)
        self.dataset['Flight_Name'] = flightNames 
    
    def fix_stop_col(self):
        stops = []
        for row in dataset['Stops']:
            row = row.replace('�', '-')
            stops.append(row)
        self.dataset['Stops'] = stops

    def fix_date_col(self):    
        self.dataset["Departure Date"] = pd.to_datetime(dataset['Departure Date'])
        self.dataset["Return Date"] = pd.to_datetime(dataset['Return Date'])

    def fix_price_col(self):
        price = []
        for row in dataset['Price']:
            row = int(row.replace('$', '').replace(',', ''))
            price.append(row)
        self.dataset['Price'] = price
    
    def save_dataset(self):
        self.dataset.to_csv(f"Supported_Data/Cleaned_{self.dataset_type}_dataset.csv", sep=',', index=False, encoding='utf-8')
    
    def start_cleaning(self):
        try:
            print("#"*50)
            print("Droping Columns ....")
            self.drop_col()
            print("Fixing Fligh Column Values ....")
            self.fix_Name_col()
            print("Fixing Stop Column Values ....")
            self.fix_stop_col()
            print("Changing Date Format Style ....")
            self.fix_date_col()
            if self.dataset_type == "Train":
                print("Converting Price To int ....")
                self.fix_price_col()
            print("Trying to save Datset (csv file) ....")
            self.save_dataset()
            print("#"*50)
            return self.dataset, True
        except Exception as e:
            print("-"*50)
            print(e)
            return self.dataset, False
              

import sys


dataset_name = sys.argv[1]
dataset_type = sys.argv[2]

dataset = pd.read_csv(f'{dataset_name}.csv')
dataset_type = dataset_type

# dataset = pd.read_csv('Dataset_test.csv')
# dataset_type = "Testing"

drop_cols = ['OriginCode', 'DestinationCode', 'Destination_Loc', 'Origin_Loc', 'Img_URL', 'Country', 'FLight_Path']


cleaning = cleaning(dataset, drop_cols, dataset_type)
dataset, status = cleaning.start_cleaning()
if not status:
    print("\nERROR IN CLEANING DATASET ....")
elif(status):
    cleaning.save_dataset()
    print("\nDATASET HAS BEEN CLEANED SUCCESSFULLY !)")
    print("DATASET HAS BEEN SAVED .....)")

