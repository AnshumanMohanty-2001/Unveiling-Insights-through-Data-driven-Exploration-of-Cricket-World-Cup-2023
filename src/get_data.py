from utils.utils import save_df, get_soup, segregate_odd_even_indices
import pandas as pd


def get_points_table(link):
    """
    function to return the points table from group stages
    :param link: url containing points table
    :return: None
    """
    columns_points_table, final_records = [[] for _ in range(2)]

    # Web scraping contents using beautiful soup
    soup = get_soup(link)

    # getting table columns
    for tag in soup.find_all('tr', class_='cb-srs-gray-strip'):
        for inner_tag in tag:
            if inner_tag.text.strip() != '':
                columns_points_table.append(inner_tag.text.strip())

    # getting table contents
    odd_tr_count = 0
    for tag in soup.find('tbody'):
        if odd_tr_count % 2 == 0:
            country_record = []
            for inner_tag in tag:
                if inner_tag.text.strip() != '':
                    country_record.append(inner_tag.text.strip())
                    continue
                final_records.append(country_record)
        odd_tr_count += 1

    # save dataframes
    save_df(columns_points_table, final_records, 'raw', 'points_table')


def get_match_schedule_results(link):
    """
    function to return the match schedule and results
    :param link: url containing match schedule and results
    :return: None
    """
    column_matches, final_records = [[] for _ in range(2)]

    # Web scraping contents using beautiful soup
    soup = get_soup(link)

    # getting table columns
    for tag in soup.find('thead', class_='ds-bg-fill-content-alternate ds-text-left'):
        for inner_tag in tag:
            column_matches.append(inner_tag.text)

    # getting table rows
    for tr_content in soup.find('tbody'):
        td_elements = tr_content.find_all('td')

        match_record = []
        for td_content in td_elements[:-1]:
            match_record.append(td_content.text)

        # getting the link to match scorecard
        match_record.append(td_elements[-1].find('a').get('href'))
        final_records.append(match_record)

    # save dataframes
    save_df(column_matches, final_records, 'raw', 'match_schedule_results')


def get_match_facts(file_path):
    """
    Function to return general match facts
    :param file_path: csv file containing match schedule and results
    :return: None
    """
    # Read the dataframe containing match schedule and results
    df = pd.read_csv(file_path)

    team_scores, man_of_match, pp_1, pp_2, pp_3, team_1_pp_1, team_1_pp_2, team_1_pp_3, \
        team_2_pp_1, team_2_pp_2, team_2_pp_3, toss, play_time, final_records = \
        [[] for _ in range(14)]

    match_num = 1  # match counter

    # loop over the match links
    for link in df['Scorecard']:

        # Web scraping contents using beautiful soup
        soup = get_soup(root_link + link)

        # Team scores
        for inner_div in soup.find('div',
                                   class_='ds-flex ds-flex-col ds-mt-3 '
                                          'md:ds-mt-0 ds-mt-0 ds-mb-1'):
            count = 0
            for inner_tag in inner_div:
                count += 1

                if count == 1:
                    continue

                team_scores.append(inner_tag.text)

        # Man of Match
        tag = soup.find('span',
                        class_='ds-text-tight-m ds-font-medium ds-text-typo '
                               'ds-underline ds-decoration-ui-stroke '
                               'hover:ds-text-typo-primary '
                               'hover:ds-decoration-ui-stroke-primary '
                               'ds-block ds-cursor-pointer')
        man_of_match.append(tag.text)

        # Powerplay_scores
        for tag in soup.find_all('ul', class_='ds-text-tight-s '
                                              'ds-font-regular '
                                              'ds-list-disc ds-pt-2 '
                                              'ds-px-4 ds-mb-4'):
            flag = False
            for inner_div in tag:
                if 'Powerplay 1:' in inner_div.text:
                    pp_1.append(inner_div.text)
                elif 'Powerplay 2:' in inner_div.text:
                    pp_2.append(inner_div.text)
                elif 'Powerplay 3:' in inner_div.text:
                    pp_3.append(inner_div.text)
                    flag = True

            if not flag and match_num != 36:  # Match 36 has redundant li
                pp_3.append('NA')
        match_num += 1

        # Toss results and playing time
        for tbody in soup.find('table', class_='ds-w-full ds-table '
                                               'ds-table-sm ds-table-auto'):
            for tr_element in tbody:
                if 'Toss' in tr_element.text:
                    toss.append(tr_element.text)
                elif 'Hours of play (local time)' in tr_element.text:
                    play_time.append(tr_element.text)

    # Segregating odd and even indices of lists for both teams
    team_1_scores, team_2_scores = segregate_odd_even_indices(team_scores)
    team_1_pp_1, team_2_pp_1 = segregate_odd_even_indices(pp_1)
    team_1_pp_2, team_2_pp_2 = segregate_odd_even_indices(pp_2)
    team_1_pp_3, team_2_pp_3 = segregate_odd_even_indices(pp_3)

    # getting table rows
    columns = ['Time', 'Toss', 'Team 1 Score', 'Team 2 Score',
               'Team 1 PP-1 Score', 'Team 1 PP-2 Score',
               'Team 1 PP-3 Score', 'Team 2 PP-1 Score',
               'Team 2 PP-2 Score', 'Team 2 PP-3 Score', 'MOM']

    # getting table columns
    for i in range(len(toss)):
        final_records.append(
            [play_time[i], toss[i], team_1_scores[i], team_2_scores[i],
             team_1_pp_1[i], team_1_pp_2[i], team_1_pp_3[i],
             team_2_pp_1[i], team_2_pp_2[i], team_2_pp_3[i], man_of_match[i]])

    # save dataframes
    save_df(columns, final_records, 'raw', 'match_facts')


if __name__ == '__main__':
    # root link
    root_link = 'https://www.espncricinfo.com/'

    # retrieve points table raw data
    points_table_url = ('https://www.cricbuzz.com/cricket-series/6732/'
                        'icc-cricket-world-cup-2023/points-table')
    get_points_table(points_table_url)

    # retrieve matches table raw data
    matches_url = ('https://www.espncricinfo.com/records/tournament/'
                   'team-match-results/icc-cricket-world-cup-2023-24-15338')
    get_match_schedule_results(matches_url)

    # retrieve general match facts
    get_match_facts('../data/raw/match_schedule_results.csv')
