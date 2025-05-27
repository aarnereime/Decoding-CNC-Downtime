from pygnos.client_factory import get_project_client as gpc
import plotly.express as px
import pandas as pd
import numpy as np


class PartDurationPlot:

    def __init__(self, df, asset: str):
        self.df = df
        self.asset = asset
        
        self.shdr_program = f'{asset}_shdr_program'
        self.machine_state = f'Machine_state_{asset}'
        self.shdr_unitNum = f'{asset}_shdr_unitNum'
        self.shdr_PartCountAct = f'{asset}_shdr_PartCountAct'
        
        
    def format_dataframe(self) -> pd.DataFrame: 
        df = self.df.copy(deep=True)
        
        df[self.shdr_program].replace('UNAVAILABLE', np.nan, inplace=True)
        df[self.shdr_program].replace('', np.nan, inplace=True)
        df[self.shdr_PartCountAct].replace('UNAVAILABLE', np.nan, inplace=True)
        
        df.ffill(inplace=True)
        
        df['part'] = (df[self.shdr_PartCountAct] != df[self.shdr_PartCountAct].shift()).cumsum()
        df['duration'] = df['timestamp'].shift(-1) - df['timestamp']
        
        part_removable = df.groupby('part').filter(lambda x: len(x) < 2)
        df = df.drop(part_removable.index)
        
        # df = df[df[self.machine_state] != 'PROGRAM STOP']
        df['cumulative_duration'] = df.groupby('part')['duration'].cumsum()
        df['cumulative_duration'] = df['cumulative_duration'].dt.total_seconds() / 60

        return df
    
    
    def make_plot(self):
        self.df = self.format_dataframe()
        self.df.to_csv('part_duration.csv', index=False)
        
        fig = px.line(self.df, x='timestamp', y='cumulative_duration', color='part', hover_data=[self.shdr_program, self.machine_state, self.shdr_unitNum])
        fig.update_layout(title=f'Part duration for {self.asset}', xaxis_title='Timestamp', yaxis_title='Cumulative duration (minutes)', template='plotly_dark')
        fig.show()        