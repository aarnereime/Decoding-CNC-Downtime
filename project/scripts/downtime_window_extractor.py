from project.scripts.cnc_data_retriever import DataRetriever
from project.config.root_dir import ROOT_DIR
from project.config.data_label_config import CLIENT

import pandas as pd
import numpy as np
import os
        

DOWNTIME_LABELS_PATH = os.path.join(ROOT_DIR, 'project/data', 'downtime_labels.csv')
DOWNTIME_WINDOW_SEQUENCES_PATH = os.path.join(ROOT_DIR, 'project/data', 'downtime_window_sequences.csv')

DOWNTIME_STATES = {
    'IDLE',
    'OPTIONAL STOP',
    'PROGRAM STOP',
    'M0',
    'ATC STOPPED',
    'FEEDHOLD',
    'MDI MODE',
    'MANUAL MODE', 
    'ALARM',
    'ESTOP',
    'POWER OFF',
}


class DowntimeWindowExtractor:
    
    def __init__(self, common_features: list[str]):
        self.cnc_data_retriever = DataRetriever(CLIENT)
        self.common_features = common_features  # Common features across all machines used for downtime detection
        
        self.timestamp_label_data = pd.read_csv(DOWNTIME_LABELS_PATH, parse_dates=['timestamp'])
        self.processed_data, self.unprocessed_data = self.divide_data()
        print('Number of processed downtime windows:', len(self.processed_data))
        print('Number of unprocessed downtime windows:', len(self.unprocessed_data))
        
        self.number_of_timestamps = len(self.unprocessed_data)
        self.extracted_window_counter = 1  # Used to track the number of extracted downtime windows   
        self.group_counter = self.get_last_known_group_id()
        
        self.duplicate_window_set = self.setup_duplicated_window_set()
        
    
    def divide_data(self):
        processed_data = self.timestamp_label_data[self.timestamp_label_data['processed_downtime_window'] == True].copy()
        unprocessed_data = self.timestamp_label_data[self.timestamp_label_data['processed_downtime_window'] == False].copy()
        return processed_data, unprocessed_data
    
    
    def get_last_known_group_id(self):
        if os.path.exists(DOWNTIME_WINDOW_SEQUENCES_PATH):
            df = pd.read_csv(DOWNTIME_WINDOW_SEQUENCES_PATH)
            return df['group_id'].max() + 1
        return 0
    
    
    def setup_duplicated_window_set(self):
        duplicate_set = set()
        if os.path.exists(DOWNTIME_WINDOW_SEQUENCES_PATH):
            df = pd.read_csv(DOWNTIME_WINDOW_SEQUENCES_PATH, parse_dates=['timestamp'])
            grouped_df = df.groupby(['group_id'])
            for group_id, group_df in grouped_df:
                # For each group, add the min and max timestamp to the set
                min_timestamp = pd.Timestamp(group_df['timestamp'].min())
                max_timestamp = pd.Timestamp(group_df['timestamp'].max())
                machine_id = group_df['machine_id'].values[0]
                duplicate_set.add((min_timestamp, max_timestamp, machine_id))
                
        return duplicate_set
    
        
    def initialize_output_csv(self):
        if not os.path.exists(DOWNTIME_WINDOW_SEQUENCES_PATH):
            with open(DOWNTIME_WINDOW_SEQUENCES_PATH, 'w') as _:
                pass
            
    
    def append_dataframe_to_csv(self, df: pd.DataFrame):
        # Checks if the file exists and is empty
        file_is_empty = os.stat(DOWNTIME_WINDOW_SEQUENCES_PATH).st_size == 0
        
        if file_is_empty:
            df.to_csv(DOWNTIME_WINDOW_SEQUENCES_PATH, mode='w', header=True, index=False)
        else:
            # If not empty, ensure column consistency
            existing_df = pd.read_csv(DOWNTIME_WINDOW_SEQUENCES_PATH, nrows=1)  # Read only headers
            existing_columns = existing_df.columns

            # Aligns the new dataframe to the existing columns
            df = df.reindex(columns=existing_columns)  # This ensures new data matches the original headers
            df.to_csv(DOWNTIME_WINDOW_SEQUENCES_PATH, mode='a', header=False, index=False)
    
        
    def format_cnc_dataframe(self, df: pd.DataFrame, machine_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame: 
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)
        
        df.replace('UNAVAILABLE', np.nan, inplace=True)
        df.ffill(inplace=True)
        
        df = df[df['timestamp'] >= start_date]
        df = df[df['timestamp'] <= end_date]
        df.reset_index(drop=True, inplace=True)
        
        df['status'] = np.where(df[f'Machine_state_{machine_id}'].isin(DOWNTIME_STATES), 'downtime', 'uptime')
        return df
        
        
    def update_common_features(self, machine_id: str):
        return [feature.replace('machine', machine_id) for feature in self.common_features]
        
    
    def get_cnc_dataframe(self, machine_id: str, timestamp: pd.Timestamp):
        start_date = timestamp - pd.Timedelta(hours=12)  #  Is set quite high to ensure we get all the data as some values are 'UNAVAILABLE'
        end_date = timestamp + pd.Timedelta(hours=2)
        
        timeseries = self.update_common_features(machine_id)
        df = self.cnc_data_retriever.retrieve_data(machine_id, start_date, end_date, timeseries, include_outside_points=True)
        return df, start_date, end_date
        
        
    def expand_downtime_window(self, df: pd.DataFrame, downtime_timestamp_index: int) -> pd.DataFrame:
        start_index = downtime_timestamp_index
        end_index = downtime_timestamp_index
        
        # First we want to find the start and end index of the downtime window
        while start_index > 0 and df.loc[start_index-1, 'status'] == 'downtime':
            start_index -= 1
            
        while end_index < len(df) - 1 and df.loc[end_index+1, 'status'] == 'downtime':
            end_index += 1
        
        return df[start_index:end_index+1].copy()
    
    
    def is_duplicate_window(self, df: pd.DataFrame, machine_id: str) -> bool:
        min_timestamp = df['timestamp'].min()
        max_timestamp = df['timestamp'].max()
        key = (min_timestamp, max_timestamp, machine_id)
        if key in self.duplicate_window_set:
            return True
        self.duplicate_window_set.add(key)
        return False
        
    
    def extract_downtime_windows(self, machine_id: str, timestamp: pd.Timestamp, label: str):
        df, start_date, end_date = self.get_cnc_dataframe(machine_id, timestamp)
        df = self.format_cnc_dataframe(df, machine_id, start_date, end_date)
        
        downtime_row =  df.loc[df['timestamp'] == timestamp]
        if downtime_row.empty:
            print(f'No downtime found at timestamp: {timestamp}')
            # find the closest timestamp to the specified timestamp (sometimes the timestamp is millisecond off)
            downtime_row = df.iloc[(df['timestamp'] - timestamp).abs().argsort()[:1]]
            if downtime_row['status'].values[0] == 'uptime':
                print(f'No downtime found at timestamp: {timestamp}')
                self.number_of_timestamps -= 1
                return
            
        downtime_index = downtime_row.index[0]
        downtime_window = self.expand_downtime_window(df, downtime_index)
        
        if self.is_duplicate_window(downtime_window, machine_id):
            print('Downtime window is a duplicate. Skipping extraction...')
            self.number_of_timestamps -= 1
            return
        
        if len(downtime_window) > 20_000:
            print('Downtime window is too long. Skipping extraction...')
            self.number_of_timestamps -= 1
            return
        
        downtime_window.columns = [col.replace(machine_id, 'machine') for col in downtime_window.columns]
        downtime_window['label'] = label
        downtime_window['group_id'] = self.group_counter
        downtime_window['machine_id'] = machine_id
        self.append_dataframe_to_csv(downtime_window)
        
        print(f'Extracted {self.extracted_window_counter} of {self.number_of_timestamps} downtime windows', end='\r')
        self.extracted_window_counter += 1
        self.group_counter += 1
            
        
    def make_sample_label_dataframe(self):
        self.initialize_output_csv()
        
        if self.unprocessed_data.empty:
            print('All timestamps have been processed')
            return
        
        print('Number of timetamps to extract downtime windows from:', self.number_of_timestamps)
        self.unprocessed_data.apply(lambda x: self.extract_downtime_windows(x['machine_id'], x['timestamp'], x['label']), axis=1)  
    
        # Mark unprocessed entries as processed and save updates
        self.unprocessed_data['processed_downtime_window'] = True
        concated_data = pd.concat([self.processed_data, self.unprocessed_data])  
        concated_data.to_csv(DOWNTIME_LABELS_PATH, index=False)        