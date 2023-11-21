import requests
from bs4 import BeautifulSoup
import pandas as pd


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


# links for team records, individual batting and bowling stats
def get_links(link, input_str):
    """
    Function to retrieve links for team, individual and bowling stats
    :param link: main link
    :param input_str: Record type
    :return: list of links
    """

    # Web scraping contents using beautiful soup
    soup = get_soup(link)

    # list of links
    links = []

    # getting table columns
    for tag in soup.find_all('div',
                             class_='ds-w-full ds-bg-fill-content-prime '
                                    'ds-overflow-hidden ds-rounded-xl '
                                    'ds-border ds-border-line ds-mb-2'):

        for inner_tag in tag:

            if inner_tag.text.strip() == input_str:
                link_container = inner_tag.nextSibling

                for li_element in link_container.find_all('li'):
                    links.append(li_element.find('a').get('href'))

    return links
