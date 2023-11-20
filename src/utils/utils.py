import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


def get_soup(url):
    """
    Function to return contents after web scraping
    :param url: url to web scrape
    :return: soup contents
    """
    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    return soup


def save_df(columns, records, df_folder, name):
    """
    Function to create and save csv files
    :param columns: list of column names
    :param records: list of records
    :param df_folder: folder to be saved
    :param name: Filename
    :return: None
    """
    df = pd.DataFrame(records, columns=columns)
    df.to_csv(f'../data/{df_folder}/{name}.csv', index=False)


def segregate_odd_even_indices(main_list):
    """
    Function to separate a list in two separate lists with
    elements at odd and even indices
    :param main_list: list to be segregated
    :return: two lists with odd and even indexed elements
    """
    return main_list[::2], main_list[1::2]
