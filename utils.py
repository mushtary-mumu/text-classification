import os
import pickle
from urllib.request import urlretrieve
from zipfile import ZipFile
import yaml

def read_yaml(path):
    """Read a YAML file and return the contents as a dictionary.

    Parameters:
    -----------
    path: str
        The path to the YAML file.

    Returns:
    --------
    out: dict
        The contents of the YAML file as a dictionary.

    """
    with open(path) as stream:
        try:
            out = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            info = (
                "The YAML file could not be loaded. Please check that the path points "
                "to a valid YAML file."
            )
            raise ValueError(info) from error
    return out

def download_file(url, dest_folder, filename):
    """Function to download a file from a URL.

    Parameters:
    -----------
    url: str
        The URL to download the file from.
    dest_folder: str
        The path to the folder where the file should be saved.
    filename: str
        The name of the file.

    Returns:
    --------
    file_path: str
        The path to the downloaded file.

    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    file_path = os.path.join(dest_folder, filename)

    urlretrieve(url=url, filename=file_path)

    return file_path