import json
import time


def get_run_identification():
    try:
        with open('/valohai/config/execution.json') as f:
            exec_details = json.load(f)
        project_name = exec_details['valohai.project-name'].split('/')[1]
        exec_id = exec_details['valohai.execution-id']
    except FileNotFoundError:
        project_name = 'test'
        exec_id = str(int(time.time()))
    return project_name, exec_id
