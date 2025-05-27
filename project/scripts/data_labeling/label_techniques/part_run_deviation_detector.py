from project.scripts.data_labeling.label_techniques.technique_base import TechniqueBase
from datetime import datetime
import pandas as pd
import numpy as np
import random


class PartRunDeviationDetector(TechniqueBase):
    
    def __init__(self, client, machine_ext_id: str):
        self.machine_ext_id = machine_ext_id
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime.now()
        super().__init__(client, self.start_date, self.end_date, monthly_range=6)
        
        self.shdr_program = f'{machine_ext_id}_shdr_program'
        self.machine_state = f'Machine_state_{machine_ext_id}'
        self.shdr_unitNum = f'{machine_ext_id}_shdr_unitNum'
        self.shdr_PartCountAct = f'{machine_ext_id}_shdr_PartCountAct'
    
    
    def get_neccecary_timeseries(self) -> list:
        return [
            self.shdr_program, 
            self.machine_state, 
            self.shdr_unitNum, 
            self.shdr_PartCountAct
        ]
    
    
    def check_if_dataframe_is_valid(self, df: pd.DataFrame) -> bool:
        """
        If the program or unit number columns are empty, the dataframe is not valid.
        """
        empty_columns = set(df.columns[df.isnull().all()])
        return self.shdr_program not in empty_columns and self.shdr_unitNum not in empty_columns
    
    
    def format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method is responsible for formatting the dataframe. It will frontfill and handle missing values for specific columns, 
        reset the index to make a timestamp column, and convert numerical columns to float. Lastly, it will remove all rows before
        the first program is found.
        """
        frontfill_cols = [
            self.shdr_program, 
            self.machine_state, 
            self.shdr_unitNum, 
            self.shdr_PartCountAct
        ]
        
        df[self.shdr_program].replace(['UNAVAILABLE', ''], np.nan, inplace=True)
        df[self.shdr_PartCountAct].replace('UNAVAILABLE', np.nan, inplace=True)
        df[self.shdr_unitNum].replace('UNAVAILABLE', np.nan, inplace=True)
        
        df[frontfill_cols] = df[frontfill_cols].ffill()
        
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'timestamp'}, inplace=True)
        
        # df = df[(df['timestamp'] >= self.start_date) & (df['timestamp'] <= self.end_date)]
        
        numerical_cols = [self.shdr_unitNum, self.shdr_PartCountAct]
        df[numerical_cols] = df[numerical_cols].astype(float)
        
        # Remove all rows before the first program is found
        idx = df[self.shdr_program].notnull().idxmax()
        df = df.iloc[idx:]
        
        return df
    
    
    def find_valid_programs(self, df: pd.DataFrame) -> list[str]:
        """
        This method is responsible for finding valid programs. A valid program is a present in the dataframe more than 1000 times,
        (this numbers comes from the fact that most programs have around 100 units/instructions (each row is a unit/instruction)) and we
        want the program to have runned more than 10 times. Some programs are also "setup" programs, which are not interesting for us and ignored
        by only looking at programs that have a unit number greater than 30 (since setup programs mostly have less unit/instructions).
        """
        valid_programs = []
        programs = df[self.shdr_program].value_counts()
        
        for program, count in programs.items():
            if count <= 1000:
                print(f'Stopping at program: {program}, since it has less than 1000 rows')
                break
            
            print(f'Program: {program} has {count} rows')
            max_unit_num = df.loc[df[self.shdr_program] == program, self.shdr_unitNum].max()
            
            if max_unit_num > 30:
                valid_programs.append(program)
            
        return valid_programs
    
    
    def fix_part_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method is responsible for fixing the part counts. Sometimes the last unit number of a part will be assigned to the next part.
        This can be fixed by comparing the unit number with the previous unit number and part count. If the unit number is not 0 and is greater
        or equal to the previous unit number, we can assume that the unit number is correct and has gone to the next unit/instruction of the program,
        therefore the part count should be the same as the previous part count. If the part count is greater than the previous part count, we know
        that the part count is wrong and need to be set to the previous part count.
        """
        previous_row = df.iloc[0]
        previous_unit_num = previous_row[self.shdr_unitNum]
        previous_part_count = previous_row[self.shdr_PartCountAct]
        previous_idx = previous_row.name
        
        for current_idx, current_row in df.iloc[1:].iterrows():
            current_unit_num = current_row[self.shdr_unitNum]
            current_part_count = current_row[self.shdr_PartCountAct]
                
            if current_unit_num != 0 and current_unit_num >= previous_unit_num:
                if current_part_count > previous_part_count:
                    df.loc[current_idx, self.shdr_PartCountAct] = previous_part_count
                elif current_part_count < previous_part_count:
                    df.loc[previous_idx, self.shdr_PartCountAct] = current_part_count
                
            # update previous values from the dataframe (not the current row since might be modified)
            previous_unit_num = df.loc[current_idx, self.shdr_unitNum]
            previous_part_count = df.loc[current_idx, self.shdr_PartCountAct]
            previous_idx = current_idx
            
        return df
    
    
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method is responsible for removing outliers. Outliers are defined as parts that have a duration that is more than 2.5 times the median
        or less than 0.5 times the median. This is done to remove parts that have run for an abnormal amount of time.
        """
        part_duration = df.groupby('part')['duration'].sum()
        
        median_duration = part_duration.median()
        upper_bound = median_duration * 2.5
        lower_bound = median_duration * 0.5
        
        outliers = part_duration[(part_duration > upper_bound) | (part_duration < lower_bound)].index
        return df[~df['part'].isin(outliers)]
    
    
    def separate_normal_and_long_part_durations(self, df: pd.DataFrame) -> tuple[list, list]:
        """
        This method is responsible for separating normal and long part durations. A normal part duration is defined as a part that has a duration
        that is close to the median duration of all parts. A long part duration is defined as a part that has a duration that is more than 2 minutes
        longer than the median duration. The method will return a list of normal parts and a list of long parts. It also removes parts that have
        a duration that is more than 4 minutes shorter than the median duration.
        """
        part_duration = df.groupby('part')['duration'].sum()
        
        median_duration = part_duration.median()
        mode_duration = part_duration.apply(lambda x: x.floor(freq='min')).mode()
        
        difference = median_duration - mode_duration
        two_minutes = pd.Timedelta(minutes=2)
              
        if abs(difference[0]) > two_minutes:
            print(f'Part duration deviates from median by more than 2 minutes, hard to find normal part run. Skipping...')
            return None, None
        
        longer_than_median = part_duration[part_duration > median_duration + pd.Timedelta(minutes=2)].index.tolist() 
        shorter_than_median = part_duration[part_duration < median_duration - pd.Timedelta(minutes=4)].index.tolist()
        
        non_normal_part = shorter_than_median + longer_than_median
        normal_parts = part_duration[~part_duration.index.isin(non_normal_part)].index.tolist()
        
        print(f'Found {len(normal_parts)} normal parts and {len(longer_than_median)} long parts')
        
        return normal_parts, longer_than_median
    
    
    def get_comparison_number(self, normal_part_durations: list) -> int:
        if len(normal_part_durations) < 10:
            comparison_num = int(0.5 * len(normal_part_durations))
        else:
            comparison_num = 5
        return comparison_num
    
    
    def find_deviation_in_long_part_durations(self, df: pd.DataFrame, normal_part_durations: list, longer_part_durations: list) -> list[datetime]:
        deviated_datapoints = []
        
        print(f'Finding deviation in long part durations')
        for part in longer_part_durations:
            comparison_num = self.get_comparison_number(normal_part_durations)
            normal_comparison = random.sample(normal_part_durations, comparison_num)
            sub_df = df[df['part'] == part].copy(deep=True)
            
            comparison_dfs = [df[df['part'] == comp_part].copy(deep=True) for comp_part in normal_comparison]
            
            max_sub_df_unit_num = int(sub_df[self.shdr_unitNum].max())
            min_sub_df_unit_num = int(sub_df[self.shdr_unitNum].min())
            
            for unit_num in range(min_sub_df_unit_num, max_sub_df_unit_num + 1):
                sub_df_duration_sum = sub_df[sub_df[self.shdr_unitNum] == unit_num]['duration'].sum()
                
                comparison_durations = [comp_df[comp_df[self.shdr_unitNum] == unit_num]['duration'].sum() for comp_df in comparison_dfs]
                avg_comparison_duration = pd.Series(comparison_durations).mean()
                
                if pd.isna(sub_df_duration_sum) or pd.isna(avg_comparison_duration):
                    continue
                
                sub_df_states = set(sub_df[sub_df[self.shdr_unitNum] == unit_num][self.machine_state].tolist())
                comparison_states = set().union(*[comp_df[comp_df[self.shdr_unitNum] == unit_num][self.machine_state].tolist() for comp_df in comparison_dfs])
                
                if not sub_df_states or not comparison_states:
                    continue
                
                lower_bound = avg_comparison_duration - pd.Timedelta(seconds=30)
                upper_bound = avg_comparison_duration + pd.Timedelta(seconds=30)
                if lower_bound <= sub_df_duration_sum <= upper_bound:
                    continue
                
                uptime_states = {'INCYCLE', 'MASTERCAM', 'CAM CYCLE', 'MDI CYCLE'}
                remaining_states = sub_df_states - comparison_states - uptime_states
                if not remaining_states:
                    continue
                
                sub_df_deviation = sub_df[(sub_df[self.shdr_unitNum] == unit_num) & (sub_df[self.machine_state].isin(remaining_states))]
                deviated_datapoints.extend(sub_df_deviation['timestamp'].tolist())
        
        print(f'Found {len(deviated_datapoints)} deviated datapoints')         
        return deviated_datapoints
    
    
    def prepare_program_data(self, df: pd.DataFrame, program: str) -> pd.DataFrame:
        sub_df = df[df[self.shdr_program] == program].copy(deep=True)
        sub_df = self.fix_part_count(sub_df)
        sub_df['part'] = (sub_df[self.shdr_PartCountAct] != sub_df[self.shdr_PartCountAct].shift()).cumsum()
        
        sub_df['duration'] = sub_df['timestamp'].shift(-1) - sub_df['timestamp']
        sub_df = self.remove_outliers(sub_df)
        
        if sub_df['part'].nunique() < 20:
            print(f'Skipping program: {program}, since it has less than 20 unique parts')
            return None
        
        sub_df['cumulative_duration'] = sub_df.groupby('part')['duration'].cumsum()
        sub_df['cumulative_duration'] = sub_df['cumulative_duration'].dt.total_seconds() / 60
        
        return sub_df
    
    
    def find_unplanned_downtime(self, df: pd.DataFrame, valid_programs: list[str]) -> list[tuple]:
        data_label_pairs = []
        
        for program in valid_programs:
            print(f'Finding unplanned downtime for program: {program}')
            sub_df = self.prepare_program_data(df, program)
            if sub_df is None:
                continue
            
            normal_part_durations, longer_part_durations = self.separate_normal_and_long_part_durations(sub_df)
            if not normal_part_durations and not longer_part_durations:
                continue
            
            deviated_datapoints = self.find_deviation_in_long_part_durations(sub_df, normal_part_durations, longer_part_durations)
            data_label_pairs.extend([(timestamp, 'unplanned') for timestamp in deviated_datapoints])
                
            # fig = px.scatter(sub_df, x='timestamp', y='cumulative_duration', color='deviation', hover_data=[self.shdr_program, self.machine_state, self.shdr_unitNum, 'part'])
            # fig.update_layout(title=f'Part duration for {self.machine_ext_id}', xaxis_title='Timestamp', yaxis_title='Cumulative duration (minutes)', template='plotly_dark')
            # fig.show()
            
        return data_label_pairs
    
    
    def handle(self) -> list[tuple]:
        """
        This method is responsible for handling the entire technique. It will retrieve the neccecary data, format it, find valid programs,
        and find unplanned downtime. The method will return a list of tuples where each tuple contains the following:
        (timestamp , label), where the first timestamp is the start time of the label, the second timestamp is the end time
        """
        data_label_pairs = []
        timeseries = self.get_neccecary_timeseries()
        
        for df, start, end in self.get_dataframe(self.machine_ext_id, timeseries):
            if self.check_if_dataframe_is_valid(df):
                print(f'Found valid data for machine: {self.machine_ext_id} between {start} and {end}')
                df = self.format_dataframe(df)
                valid_programs = self.find_valid_programs(df)
                new_data_label_pairs = self.find_unplanned_downtime(df, valid_programs)
                
                data_label_pairs.extend(new_data_label_pairs)
            else:
                continue 
        
        return data_label_pairs