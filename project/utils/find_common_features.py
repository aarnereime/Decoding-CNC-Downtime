from project.config.data_label_config import CLIENT, MACHINE_EXTERNAL_ID

class FindCommonFeatures:
    
    def remove_machine_id_from_timeseries_string(self, timeseries: list[str], machine_id: str):
        return [i.replace(machine_id, 'machine') for i in timeseries]
    
    
    def get_machine_specific_timeseries(self, machine_id: str):
        res = CLIENT.time_series.list(asset_external_ids=[machine_id], limit=None)
        timeseries = [i.external_id for i in res]
        timeseries = self.remove_machine_id_from_timeseries_string(timeseries, machine_id)
        return timeseries
    
    
    def filter_timeseries(self, timeseries: list):
        # Some timeseries are duplicate, of each other, one numeric and one string type, we only want the numeric type
        set_timeseries = set(timeseries)
        
        for ts in timeseries:
            if '_Numeric' in ts:
                temp = ts.replace('_Numeric', '')
                if temp in set_timeseries:
                    set_timeseries.remove(temp)
                    
        # Hand picked timeseries to remove
        timeseries_to_remove = [
            'machine_shdr_avail',
            'Shifts_machine',
            'machine_d1_asset_rem',
            'machine_avail',
            'Heartbeat_machine_prod',
            'machine_shdr_estop',
            'machine_shdr_connection',
            'machine_d1_asset_chg',
            'machine_shdr_mode',
        ]
        for ts in timeseries_to_remove:
            if ts in set_timeseries:
                set_timeseries.remove(ts)
                    
        return list(set_timeseries)
        
        
    def  get_common_features(self):
        common_features = []
        
        for machine_id in MACHINE_EXTERNAL_ID:
            timeseries = self.get_machine_specific_timeseries(machine_id)
            common_features.append(timeseries)
            
        common_features = set.intersection(*map(set, common_features))
        common_features = self.filter_timeseries(list(common_features))
        return common_features