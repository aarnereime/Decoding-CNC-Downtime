from project.scripts.data_labeling.label_techniques.technique_base import TechniqueBase
from datetime import datetime
import pandas as pd


class MissingTool(TechniqueBase):
    
    def __init__(self, client, machine_ext_id: str):
        self.machine_ext_id = machine_ext_id
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime.now()
        super().__init__(client, self.start_date, self.end_date, monthly_range=6)
        
        self.machine_state = f'Machine_state_{machine_ext_id}'
        
    
    def check_if_dataframe_is_valid(self, df: pd.DataFrame) -> bool:
        """
        If the machine state is empty, the dataframe is not valid.
        """
        empty_columns = set(df.columns[df.isnull().all()])
        return self.machine_state not in empty_columns
    
    
    def format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)
        
        return df
    
    
    def add_events_to_dataframe(self, df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
        start_time_in_ms = int(start.timestamp() * 1000)
        end_time_in_ms = int(end.timestamp() * 1000)

        events = self.client.events.list(
            asset_external_ids=[self.machine_ext_id], 
            start_time={'min': start_time_in_ms, 'max': end_time_in_ms},
            limit=None
        ).to_pandas()
        
        events.rename(columns={'start_time': 'timestamp'}, inplace=True)
        events = events[['timestamp', 'type', 'subtype', 'description']]
        
        combined = pd.merge(df, events, on='timestamp', how='outer')
        combined.sort_values(by='timestamp', inplace=True)
        combined[self.machine_state].ffill(inplace=True)
        combined.reset_index(drop=True, inplace=True)
        
        return combined
    
    
    def format_description_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df['description'] = df['description'].str.strip()
        df['description'] = df['description'].str.upper()
        return df
    
    
    def add_label_to_downtime(self, planned_downtime: list[datetime]) -> list[datetime]:
        return [(timestamp, 'unplanned') for timestamp in planned_downtime]
    
    
    def find_downtime_missing_tool(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_tool_description = 'VERKTØY KAN IKKE FINNES'
        missing_tool_df = df[df['description'] == missing_tool_description]
        missing_tool_df = missing_tool_df[missing_tool_df[self.machine_state].notnull()]
        
        missing_tool_df.reset_index(inplace=True)
        missing_tool_df.rename(columns={'index': 'original_df_index'}, inplace=True)
        
        if not missing_tool_df.empty:
            missing_tool_df['group'] = missing_tool_df['original_df_index'].diff().ne(1).cumsum()
            
            missing_tool_df = missing_tool_df.groupby('group')['original_df_index'].max().reset_index()
            next_state_indexes = [index + 1 for index in missing_tool_df['original_df_index']]
            
            next_state_df = df.loc[next_state_indexes]
            missing_tool_downtime_timestamps = next_state_df.loc[next_state_df[self.machine_state] == 'IDLE', 'timestamp'].tolist()
            planned_downtime_label = self.add_label_to_downtime(missing_tool_downtime_timestamps)
            
            return planned_downtime_label
            
        return None
    
        
    def handle(self):
        data_label_pairs = []
        timeseries = [self.machine_state]
        
        for df, start, end in self.get_dataframe(self.machine_ext_id, timeseries):
            if self.check_if_dataframe_is_valid(df):
                df = self.format_dataframe(df)
                df = self.add_events_to_dataframe(df, start, end)
                df = self.format_description_column(df)
                
                planned_downtime = self.find_downtime_missing_tool(df)
                if planned_downtime is None:
                    continue
                
                data_label_pairs.extend(planned_downtime)
                
        print(f'Found {len(data_label_pairs)} missing tool downtime for machine: {self.machine_ext_id}')
        
        return data_label_pairs