from project.config.data_label_config import CLIENT, MACHINE_EXTERNAL_ID
from project.config.root_dir import ROOT_DIR
from project.scripts.cnc_data_retriever import DataRetriever
from project.scripts.data_labeling.label_techniques.part_run_deviation_detector import PartRunDeviationDetector
from project.scripts.data_labeling.label_techniques.missing_tool import MissingTool
from project.scripts.data_labeling.label_techniques.overload import Overlast
from project.scripts.data_labeling.label_techniques.program_comment import ProgramComment

import pandas as pd
import os


DOWNTIME_LABELS_PATH = os.path.join(ROOT_DIR, 'project/data', 'downtime_labels.csv')


class DataLabeling:
    """
    This class is responsible for labeling the data that is retrieved from the CNC machines. The labeling will be stored
    in a dictionary where the key is the machine name and the value is a list of tuples where each tuple contains the following:
    ((timestamp, timestamp), label), where the first timestamp is the start time of the label, the second timestamp is the end time
    of the label, and the label is the label of the data. This is primarily done to computationally reduce the amount of data that
    needs to be stored in memory.
    """

    def __init__(self):
        self.setup_csv() # Setup the csv file if it does not exist
        self.existing_data: set[tuple] = self.load_data_label_pairs() # Load the existing data from the csv file
        self.cnc_data_retriever = DataRetriever(CLIENT)
        self.rows_added = 0
        
        # Dictionary of techniques that can be used to label the data
        self.techniques = {
            'part_run_deviation_detector': PartRunDeviationDetector,
            'missing_tool': MissingTool,
            'overlast': Overlast,
            'program_comment': ProgramComment
        }
        
        
    def setup_csv(self):
        if not os.path.exists(DOWNTIME_LABELS_PATH):
            print(f'Creating new csv file: {DOWNTIME_LABELS_PATH}')
            with open(DOWNTIME_LABELS_PATH, 'w') as f:
                f.write('machine_id,timestamp,label,processed_downtime_window\n')
                
    
    def load_data_label_pairs(self):
        current_data = pd.read_csv(DOWNTIME_LABELS_PATH, usecols=['machine_id', 'timestamp', 'label'], parse_dates=['timestamp'])
        return set(current_data.itertuples(index=False, name=None))
        
                
    def make_machine_timestamp_label_set(self, new_data_label_pairs: list[tuple], machine_ext_id: str):
        return {(machine_ext_id, new_timestamp, new_label, False) for new_timestamp, new_label in new_data_label_pairs}
            
        
    def add_data_to_csv(self, machine_timestamp_label_set):
        new_values_to_add = {row for row in machine_timestamp_label_set if row[:3] not in self.existing_data}
        print(f'Adding {len(new_values_to_add)} new values to csv')
        self.rows_added += len(new_values_to_add)
        
        if new_values_to_add:
            new_rows_df = pd.DataFrame(
                list(new_values_to_add),
                columns=['machine_id', 'timestamp', 'label', 'processed_downtime_window']
            )
        
            new_rows_df.to_csv(DOWNTIME_LABELS_PATH, mode='a', header=False, index=False)
                
                
    def run_technique(self, technique_name: str):
        if technique_name not in self.techniques:
            raise ValueError(f'Technique {technique_name} not found.')
        
        technique = self.techniques[technique_name]
        
        print(f'Running technique: {technique_name}')
        
        for machine_ext_id in MACHINE_EXTERNAL_ID:
            t = technique(CLIENT, machine_ext_id)
            new_data_label_pairs = t.handle()
            machine_timestamp_label_set = self.make_machine_timestamp_label_set(new_data_label_pairs, machine_ext_id)
            print(len(machine_timestamp_label_set))
            self.add_data_to_csv(machine_timestamp_label_set)
            
        print(f'Finished running technique: {technique_name}, added {self.rows_added} new rows to csv')