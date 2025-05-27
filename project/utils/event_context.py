def convert_timetamp_to_milliseconds(timestamp):
    return timestamp.timestamp() * 1000


def get_event_context(client, machine_ext_id, datapoint):
    timestamp = datapoint.get('timestamp')
    milliseconds_timestamp = convert_timetamp_to_milliseconds(timestamp)    
    
    event_datapoints = client.events.list(
        asset_external_ids=[machine_ext_id],
        start_time={
            'min': milliseconds_timestamp,
            'max': milliseconds_timestamp
        },
        limit = 1
    )
    if not event_datapoints:
        return None
    
    event_data = event_datapoints[0]
    type_of_event = event_data.type
    
    return type_of_event.lower()