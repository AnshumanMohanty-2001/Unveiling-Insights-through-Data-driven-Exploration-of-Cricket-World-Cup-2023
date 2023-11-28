import os
import pandas as pd
import numpy as np
from src.visualize_results import get_bar, get_line_chart


# Create a directory to save the HTML file
plot_directory_path = 'results/plots/analysis_plots'
os.makedirs(plot_directory_path, exist_ok=True)


def get_team_standings(df):
    """
    function to analyze the team standings on the group stages
    :param df: points table csv
    :return: None
    """

    # get a sub dataframe for dataframe to analyze relevant columns
    sub_df_points = df[['Teams', 'Won', 'Qualification_Status']]

    # plot bar chart to analyse the points obtained by each team and their qualification status
    get_bar(sub_df_points, 'Teams', 'Won', 'Matches won by Teams',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Teams',
            'matches_won', 'teams_matches_plot', color='Qualification_Status')


def get_columnwise_records(df):
    df = pd.DataFrame(df)

    # Convert each column to a separate list
    column_lists = {column: df[column].tolist() for column in df.columns}

    return column_lists


def get_team_stats(df_match_summary, df_points):
    df_match_summary = df_match_summary[:-3]  # Include only the group stage matches

    # make a sub dataframe with relevant columns for analysis
    team_runs_per_wicket_df_team = df_match_summary[['Team 1', 'Team 1 Runs Scored','Team 1 Wickets Lost','Team 2' , 'Team 2 Runs Scored', 'Team 2 Wickets Lost']]

    # get column records in the form of dict with key as Column name
    # and value as list of team names
    dict_column_records = get_columnwise_records(team_runs_per_wicket_df_team)

    # Combine stats of Team 1 and Team 2
    teams, runs, wickets = [[] for _ in range(3)]
    for k, v in dict_column_records.items():
        if k == 'Team 1':
            teams = v + dict_column_records['Team 2']
        elif k == 'Team 1 Runs Scored':
            runs = v + dict_column_records['Team 2 Runs Scored']
        elif k == 'Team 1 Wickets Lost':
            wickets = v + dict_column_records['Team 2 Wickets Lost']

    # make a df with new columns
    df_team_stats = pd.DataFrame({'Teams': teams, 'runs': runs, 'wickets': wickets})

    # Calculate average runs per wicket for each team
    result_df_avg_runs = df_team_stats.groupby('Teams').apply(lambda x: (x['runs'].sum()) / (x['wickets'].sum())).reset_index(name='Average Runs per Wicket')

    # Sort the result DataFrame in descending order
    result_df_avg_runs = result_df_avg_runs.sort_values(by='Average Runs per Wicket', ascending=False)

    # merge qualification status on the existing dataframe
    # Merge dataframes on the 'Team' column
    final_avg_runs_per_wicket_df = pd.merge(result_df_avg_runs , df_points, on='Teams')

    # plot bar chart to analyse the average runs scored by each team and their qualification status
    get_bar(final_avg_runs_per_wicket_df, 'Teams', 'Average Runs per Wicket', 'Average Runs per Wicket for all teams',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Team Name',
            'Average Runs', 'teams_matches_average_runs_plot', color='Qualification_Status')

    # average_wickets_lost
    result_df_avg_wickets = df_team_stats.groupby('Teams').apply(
        lambda x: (x['wickets'].sum()) / len(x['wickets'])).reset_index(name='Average Wickets lost')

    # Sort the result DataFrame in descending order
    result_df_avg_wickets = result_df_avg_wickets.sort_values(by='Average Wickets lost', ascending=False).reset_index(
        drop=True)

    # merge qualification status on the existing dataframe
    # Merge dataframes on the 'Team' column
    final_result_df_avg_wickets = pd.merge(result_df_avg_wickets , df_points, on='Teams')

    # plot bar chart to analyse the average runs scored by each team and their qualification status
    get_bar(final_result_df_avg_wickets, 'Teams', 'Average Wickets lost', 'Average Wickets lost per match all teams',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Team Name',
            'Wickets Lost', 'teams_matches_average_wickets_plot', color='Qualification_Status')

    # analysis of first innings scores

    # make a sub dataframe with relevant columns for analysis
    first_innings_df = df_match_summary[['Team 1', 'Winner']]

    # Replace 'Winner' column values with 1 if equal to 'Team 1', else 0
    first_innings_df['Winner'] = np.where(first_innings_df['Winner'] == first_innings_df['Team 1'], 1, 0)




def get_teams_matchwise_trend(df_match_summary):
    """
    function to analyze the match by match performance of teams
    :param df_match_summary: match summary csv
    :return: None
    """

    # get a sub dataframe for dataframe to analyze relevant columns
    sub_df_match_summary = df_match_summary[['Team 1', 'Team 2', 'Winner']]

    # Create a new column for the team that lost
    sub_df_match_summary['Team Lost'] = sub_df_match_summary.apply(
        lambda record: record['Team 1'] if record['Winner'] != record['Team 1'] else record['Team 2'], axis=1)
    df_team_won_lost = sub_df_match_summary[['Winner', 'Team Lost']]

    # Exclude the last three records (only considering group stages for this analysis)
    df_team_won_lost = df_team_won_lost[:-3]

    # Create a dictionary with keys as team names and values as lists containing 'Win' or 'Lose'
    team_results = {}

    for index, row in df_team_won_lost.iterrows():
        winner_team = row['Winner']
        team_lost = row['Team Lost']

        if winner_team not in team_results:
            team_results[winner_team] = ['Win']
        else:
            team_results[winner_team].append('Win')

        if team_lost not in team_results:
            team_results[team_lost] = ['Lose']
        else:
            team_results[team_lost].append('Lose')

    # Replace 'Win' with 2 and 'Lose' with 0
    # (0 and 2 are the points obtained on losing and winning respectively)
    team_results_points = {team: [2 if result == 'Win' else 0 for result in results] for team, results in
                           team_results.items()}

    # Generate cumulative list for each team
    team_results_cumulative = {team: np.cumsum(results) for team, results in team_results_points.items()}

    # Convert the dictionary to a DataFrame
    df_team_matchwise_points = pd.DataFrame(
        [(team, match + 1, points) for team, results in team_results_cumulative.items() for match, points in
         enumerate(results)],
        columns=['Team', 'Match', 'Points'])

    get_line_chart(df_team_matchwise_points, 'Match', 'Points', 'Team matchwise winning trends', 'matchwise_points',
                   'Team')


def run_analysis():

    # get team standings on the points table (group stages)
    df_points = pd.read_csv('data/processed/points_table.csv')
    get_team_standings(df_points)

    # analyze matchwise trend of each team during group stages
    df_match_summary = pd.read_csv('data/processed/match_summary.csv')
    get_teams_matchwise_trend(df_match_summary)

    # get average runs per wicket for all teams
    # get average runs per wicket for all teams batting 1st and 2nd
    get_team_stats(df_match_summary, df_points)


