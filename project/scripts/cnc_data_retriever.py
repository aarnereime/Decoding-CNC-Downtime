from cognite.client import CogniteClient

import pandas as pd
       
        
class DataRetriever:
    
    def __init__(self, client: CogniteClient):
        self.client = client
    

    def get_timeseries(self, machine_ext_id: str) -> list:
        """
        Returns a list of all timeseries for the specified machine.
        """
        res = self.client.time_series.list(asset_external_ids=[machine_ext_id], limit=None)

        # Check if the response is empty
        if not res:
            raise ValueError(f'No timeseries found for machine id: {machine_ext_id}')

        timeseries = [i.external_id for i in res]
        print(f'Found {len(timeseries)} timeseries for machine id: {machine_ext_id}')

        return timeseries
    
    
    def retrieve_data(self, machine_ext_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp, timeseries: list = None, **kwargs) -> pd.DataFrame:
        """
        Generates a DataFrame using the timeseries list within the specified time range.
        """
        if timeseries is None:
            timeseries = self.get_timeseries(machine_ext_id)
        
        dataframe = self.client.time_series.data.retrieve(
            external_id=timeseries,
            start=start_date,
            end=end_date,
            limit=None,
            **kwargs
        ).to_pandas()
        return dataframe

