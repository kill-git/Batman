from batman.core.data_preprocessing import fetch_eCO2mix_data, convert_all_xls_eCO2mix_data

def fetch_data_node(destination_folder: str):
    fetch_eCO2mix_data(destination_folder)
    return "done" # signal that the fetch is done

def convert_data_node(xls_path: str, csv_path: str, _:str):
    convert_all_xls_eCO2mix_data(xls_path, csv_path)