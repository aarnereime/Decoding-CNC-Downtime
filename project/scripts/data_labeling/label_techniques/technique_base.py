from project.scripts.cnc_data_retriever import DataRetriever

import pandas as pd

class TechniqueBase:
    
    def __init__(self, client, start_date, end_date, monthly_range: int=6):
        self.client = client
        self.dr = DataRetriever(client)
        self.start_date = start_date
        self.end_date = end_date
        self.monthly_range = monthly_range
        
        
    def generate_monthly_ranges(self):
        monthly_timestamps = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')
        
        for i in range(0, len(monthly_timestamps) - 1, self.monthly_range):
            start = monthly_timestamps[i]
            if i + self.monthly_range < len(monthly_timestamps):
                end = monthly_timestamps[i + self.monthly_range]
            else:
                end = self.end_date 
            yield start, end
         
        
    def get_dataframe(self, machine_ext_id: str, timeseries: list):
        print(f'Retrieving data for machine: {machine_ext_id}')
        for start, end in self.generate_monthly_ranges():
            df = self.dr.retrieve_data(machine_ext_id, start, end, timeseries)
            yield df, start, end