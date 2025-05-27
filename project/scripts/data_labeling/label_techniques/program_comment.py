from project.scripts.data_labeling.label_techniques.technique_base import TechniqueBase
from datetime import datetime
import pandas as pd


class ProgramComment(TechniqueBase):
    
    def __init__(self, client, machine_ext_id: str):
        self.machine_ext_id = machine_ext_id
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime.now()
        super().__init__(client, self.start_date, self.end_date, monthly_range=12)
        
        self.machine_state = f'Machine_state_{machine_ext_id}'
        self.shdr_program_cmt = f'{machine_ext_id}_shdr_program_cmt'
        self.shdr_subprogram_cmt = f'{machine_ext_id}_shdr_subprogram_cmt'
        
        self.maybe_planned_program_comment = {
            'TOPCUT-BARFEEDER',
            'topcut-barfeeder',
            '4-POINT BORE/BOSS MEASUREMENT B0',
            '3-POINT BORE/BOSS MEASUREMENT B0',
            '4-Point Bore/Boss measurement B0',
            'SETTE SENTER PA KALIBRERINGSKULE',
            'SJEKK AV DIA ETTER KALIBRERING',
            '3-Point Bore/Boss measurement B0',
            'RENISHAW KALIBRERING',
            'Sette senter av kalibreringskule',
            'SETTE BAKKER',
            'PROBEKALIB',
            'MAZACONFIG',
            'BAKSEKALIB',
            'JUSTERING SENTER INV',
            'MAALE LENGDE PAA DEL',
            'RETTE BRILLE/BAKDOKKE',
            'SPISS/BRILLE RETT',
            'BRILLESJEKK',
            'MAL MAALE DIA UTV ARC',
            'SJEKK AV KULEDIAMETER ETTER KALIBRERING',
            'BAKKEPROG',
            'OPPRETTING',
            'Kalibrere Renishaw',
            'DREIING AV BAKKER',
            'BRILLESJEKK/SENTERSJEKK',
            'KALIBRERING AV RENISHAW',
        }
        
    
    def check_if_dataframe_is_valid(self, df: pd.DataFrame) -> bool:
        """
        If the program cmt is empty, the dataframe is not valid.
        """
        empty_columns = set(df.columns[df.isnull().all()])
        return self.shdr_program_cmt not in empty_columns and self.shdr_subprogram_cmt not in empty_columns
    
    
    def format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)
        
        df[self.shdr_program_cmt] = df[self.shdr_program_cmt].str.strip()
        df[self.shdr_subprogram_cmt] = df[self.shdr_subprogram_cmt].str.strip()
        
        return df
    
    
    def add_label_to_downtime(self, planned_downtime: list[datetime]) -> list[datetime]:
        return [(timestamp, 'planned') for timestamp in planned_downtime]
        
    
    def find_planned_downtime(self, df: pd.DataFrame) -> list[datetime]:
        downtime_states = {
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
        
        program_comment_df = df[df[self.shdr_program_cmt].isin(self.maybe_planned_program_comment)]
        subprogram_comment_df = df[df[self.shdr_subprogram_cmt].isin(self.maybe_planned_program_comment)]
        
        planned_program_comment = program_comment_df[program_comment_df[self.machine_state].isin(downtime_states)]['timestamp'].tolist()
        planned_subprogram_comment = subprogram_comment_df[subprogram_comment_df[self.machine_state].isin(downtime_states)]['timestamp'].tolist()
        
        planned_downtime = list(set(planned_program_comment + planned_subprogram_comment))
        print(f'Found {len(planned_downtime)} planned downtime for machine: {self.machine_ext_id}')
        
        planned_downtime_label = self.add_label_to_downtime(planned_downtime)
        return planned_downtime_label
        
        
    def handle(self):
        data_label_pairs = []
        timeseries = [self.machine_state, self.shdr_program_cmt, self.shdr_subprogram_cmt]
        
        for df, start, end in self.get_dataframe(self.machine_ext_id, timeseries):
            if self.check_if_dataframe_is_valid(df):
                print(f'Found valid data for machine: {self.machine_ext_id} between {start} and {end}')
                df = self.format_dataframe(df)
                planned_downtime = self.find_planned_downtime(df)
                data_label_pairs.extend(planned_downtime)
        
        return data_label_pairs