import missingno as msno
import matplotlib.pyplot as plt
import os


def get_msno_matrix(df, directory_path, filename):
    """
    function to generate missing value matrix
    :param df: dataframe
    :param directory_path: path of the file
    :param filename: file_name
    :return: None
    """

    # Create the missing values matrix
    msno.matrix(df)

    # base folder
    base_directory = 'results'

    # Create full paths
    full_directory_path = os.path.join(base_directory, directory_path)

    # Check and create directories
    if not os.path.exists(full_directory_path):

        os.makedirs(full_directory_path)

    # Save the visualization as an image
    output_path = os.path.join(full_directory_path, filename.split('.')[0])
    plt.savefig(output_path)
