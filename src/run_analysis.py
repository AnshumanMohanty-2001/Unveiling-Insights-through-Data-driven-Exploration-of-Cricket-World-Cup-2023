import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.visualize_results import get_bar, get_line_chart, heatmap, get_pie

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
            'matches_won', 'teams_matches_plot', color = 'Qualification_Status')


def get_columnwise_records(df):
    df = pd.DataFrame(df)

    # Convert each column to a separate list
    column_lists = {column: df[column].tolist() for column in df.columns}

    return column_lists


def get_overall_team_stats(df_match_summary, df_points):
    """
    function to analyze average runs per wicket and
    average wickets lost per match for all teams
    :param df_match_summary: match summary csv
    :param df_points: points table csv
    :return: None
    """

    df_match_summary = df_match_summary[:-3]  # Include only the group stage matches

    # make a sub dataframe with relevant columns for analysis
    team_runs_per_wicket_df_team = df_match_summary[
        ['Team 1', 'Team 1 Runs Scored', 'Team 1 Wickets Lost', 'Team 2', 'Team 2 Runs Scored', 'Team 2 Wickets Lost']]

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
    result_df_avg_runs = df_team_stats.groupby('Teams').apply(
        lambda x: (x['runs'].sum()) / (x['wickets'].sum())).reset_index(name='Average Runs per Wicket')

    # Sort the result DataFrame in descending order
    result_df_avg_runs = result_df_avg_runs.sort_values(by='Average Runs per Wicket', ascending=False)

    # merge qualification status on the existing dataframe
    # Merge dataframes on the 'Team' column
    final_avg_runs_per_wicket_df = pd.merge(result_df_avg_runs, df_points, on='Teams')

    # plot bar chart to analyse the average runs scored by each team and their qualification status
    get_bar(final_avg_runs_per_wicket_df, 'Teams', 'Average Runs per Wicket', 'Average Runs per Wicket for all teams',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Team Name',
            'Average Runs', 'teams_matches_average_runs_plot')

    # average_wickets_lost
    result_df_avg_wickets = df_team_stats.groupby('Teams').apply(
        lambda x: (x['wickets'].sum()) / len(x['wickets'])).reset_index(name='Average Wickets lost')

    # Sort the result DataFrame in descending order
    result_df_avg_wickets = result_df_avg_wickets.sort_values(by='Average Wickets lost', ascending=False).reset_index(
        drop=True)

    # merge qualification status on the existing dataframe
    # Merge dataframes on the 'Team' column
    final_result_df_avg_wickets = pd.merge(result_df_avg_wickets, df_points, on='Teams')

    # plot bar chart to analyse the average wickets lost by each team and their qualification status
    get_bar(final_result_df_avg_wickets, 'Teams', 'Average Wickets lost', 'Average Wickets lost per match all teams',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Team Name',
            'Wickets Lost', 'teams_matches_average_wickets_plot')

    # find out the correlation of the overall team stats

    # Dropping the Total matches, Pts, NRR column as it's irrelevant for this analysis
    # Dropping lost column as winning and lost are complement of each other
    final_avg_runs_per_wicket_df.drop(['Mat', 'Lost', 'Pts', 'NRR'], axis=1, inplace=True)

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Merge the avg runs per wicket df and avg wickets lost per match
    overall_team_stats_df = pd.merge(final_avg_runs_per_wicket_df, result_df_avg_wickets, on='Teams')

    # Fit and transform the Qualification_Status column
    overall_team_stats_df['Qualification_Status'] = label_encoder.fit_transform(
        overall_team_stats_df['Qualification_Status'])

    # Generate a correlation heatmap
    heatmap(overall_team_stats_df, 'overall_team_stats_heatmap', 'Correlation of Team statistics')


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


def get_first_second_innings_stats(df_match_summary, df_points):
    sub_first_innings_df = df_match_summary[
                               ['Team 1', 'Team 2', 'Team 1 Runs Scored', 'Team 1 Overs', 'Team 1 Wickets Lost',
                                'Winner']][:-3]

    # Overall 1st and 2nd batting trend (Win and losing)
    tournament_1st_2nd = sub_first_innings_df[['Team 1', 'Team 2', 'Winner']]

    first_bat_win = tournament_1st_2nd[tournament_1st_2nd['Team 1'] == tournament_1st_2nd['Winner']]
    second_bat_win = tournament_1st_2nd[tournament_1st_2nd['Team 2'] == tournament_1st_2nd['Winner']]

    overall_data = {
        'Batting': ['1st Batting', '2nd Batting'],
        'Matches_Won': [first_bat_win.shape[0], second_bat_win.shape[0]]
    }

    overall_data = pd.DataFrame(overall_data)

    # plot the pie chart
    get_pie(overall_data, 'Batting', 'Matches Won Batting 1st v/s 2nd', 'matches_won_batting_1_2')

    # get the times teams have batted first
    first_batting_count = sub_first_innings_df['Team 1'].value_counts()

    # get the total number of times teams have batted second
    second_batting_count = list(map(lambda x: 9 - x, list(sub_first_innings_df['Team 1'].value_counts())))

    batting_times_df = pd.DataFrame(list(first_batting_count.to_dict().items()), columns=['Team', '1st Batting Count'])
    batting_times_df['2nd Batting Count'] = second_batting_count

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted = pd.melt(batting_times_df, id_vars='Team', var_name='Batting Type', value_name='Count')

    # plot the bar chart to show total number of time teams have batted first v/s second
    get_bar(df_melted, 'Team', 'Count', 'Total count of teams batted 1st v/s 2nd',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Team Name',
            'Total Count', '1st_2nd_batting_counts', color='Batting Type', barmode='group')

    # Wins batting 1st and batting 2nd
    # Create a new column 'Winning_Team' based on the condition
    sub_first_innings_df['First_Winning_Team'] = (
                sub_first_innings_df['Team 1'] == sub_first_innings_df['Winner']).astype(int)

    # assigning a copy of the sub dataframe
    first_batting_df = sub_first_innings_df.copy()

    # Rename the 'Team' column to 'Team 1'
    first_batting_df = first_batting_df.rename(columns={'Team 1': 'Team'})

    # get value counts for each team
    total_counts_first = first_batting_df.groupby('Team')['First_Winning_Team'].sum().reset_index()

    first_in_wins = pd.merge(batting_times_df, total_counts_first, on='Team')

    first_in_wins.drop('2nd Batting Count', axis=1, inplace=True)

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted = pd.melt(first_in_wins, id_vars='Team', var_name='First_Winning_Team', value_name='Count')

    # plot the bar chart to show total number of time teams have batted first v/s second
    get_bar(df_melted, 'Team', 'Count', 'Teams Batting 1st Count and Total wins batting 1st',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Team Name',
            'Total Count', 'First_Batting_Wins', color='First_Winning_Team', barmode='group')

    # Create a new column 'Winning_Team' based on the condition

    sub_first_innings_df['Second_Winning_Team'] = (
                sub_first_innings_df['Team 2'] == sub_first_innings_df['Winner']).astype(int)

    # assigning a copy of the sub dataframe
    second_batting_df = sub_first_innings_df.copy()

    # Rename the 'Team' column to 'Team 1'
    second_batting_df = second_batting_df.rename(columns={'Team 2': 'Team'})

    # get value counts for each team
    total_counts_second = second_batting_df.groupby('Team')['Second_Winning_Team'].sum().reset_index()

    second_in_wins = pd.merge(batting_times_df, total_counts_second, on='Team')
    second_in_wins.drop('1st Batting Count', axis=1, inplace=True)

    # Melt the DataFrame to reshape it for side-by-side bar chart
    df_melted = pd.melt(second_in_wins, id_vars='Team', var_name='Second_Winning_Team', value_name='Count')

    # plot the bar chart to show total number of time teams have batted first v/s second
    get_bar(df_melted, 'Team', 'Count', 'Teams Batting 2nd Count and Total wins batting 2nd',
            {'Q': 'limegreen', 'E': 'greenyellow'}, 'Team Name',
            'Total Count', 'Second_batting_Wins', color='Second_Winning_Team', barmode='group')

    # frequency of teams getting bowled out before 50 overs

    # Filter the DataFrame based on the condition
    bowled_out_df = sub_first_innings_df[
        (sub_first_innings_df['Team 1 Overs'] < 50) & (sub_first_innings_df['Team 1 Wickets Lost'] == 10)]

    # Count occurrences and handle teams not bowled out
    team_counts = bowled_out_df['Team 1'].value_counts().reindex(df_match_summary['Team 1'].unique(), fill_value=0)

    # Sort counts in descending order
    team_counts = team_counts.sort_values(ascending=False)

    bowled_out_df = pd.DataFrame(list(team_counts.to_dict().items()), columns=['Team', 'Bowled Out Times'])

    # plot the bar chart to show total number of time teams have gotten bowled out
    get_bar(bowled_out_df, 'Team', 'Bowled Out Times',
            'Frequency of Teams getting bowled out before playing 50 overs (1st innings)',
            None, 'Team Name',
            'Bowled out count', 'Bowled_out_frequency')

    # %age of teams winning and lossing getting bowled out for less than 300 in lesser than 50 overs

    # getting a df copy
    bowled_match_winning_df = sub_first_innings_df.copy()

    # Filter teams with overs less than 50, runs less than 300 and lost 10 wickets (to remove the ambiguity of playing less overs due to DLS)
    bowled_match_winning_df = bowled_match_winning_df[
        (bowled_match_winning_df['Team 1 Overs'] < 50) & (bowled_match_winning_df['Team 1 Wickets Lost'] == 10) & (
                    bowled_match_winning_df['Team 1 Runs Scored'] < 300)]

    # Create a new column 'Winning Team' with 1 if 'Team' equals 'Winner', else 0
    bowled_match_winning_df['Winning Team'] = np.where(
        bowled_match_winning_df['Team 1'] == bowled_match_winning_df['Winner'], 1, 0)

    # Map values in 'Winning Team' column to labels
    bowled_match_winning_df['Result'] = bowled_match_winning_df['Winning Team'].map({0: 'Losing', 1: 'Winning'})

    # plot the pie chart
    get_pie(bowled_match_winning_df, 'Result',
            'Winning and Losing %age of teams getting bowled out less than 300 before playing 50 overs',
            'less_than_300_less_than_50')

    bowled_teams_before_df = bowled_match_winning_df.copy()

    bowled_teams_before_df = bowled_teams_before_df[['Team 1', 'Winner']]

    # Get total value counts where 'Team 1' is equal to 'Winner'
    total_lost_counts = bowled_teams_before_df[
        bowled_teams_before_df['Team 1'] != bowled_teams_before_df['Winner']].groupby('Team 1').size().reset_index(
        name='Total Losses')

    # Display the total value counts
    total_lost_counts = total_lost_counts.sort_values(by='Total Losses', ascending=False)

    # plot the bar chart to show total number of time teams have lost after getting bowled out for less than 300 before 50 overs
    get_bar(total_lost_counts, 'Team 1', 'Total Losses',
            'Teams losing after getting bowled out before playing 50 overs and scoring less than 300 (1st innings)',
            None, 'Team Name',
            'Losses count', 'Teams_Bowled_out_losses')

    # %age of teams winning after scoring 300 in th first innings

    # getting a df copy
    batting_match_winning_df = sub_first_innings_df.copy()

    # Filter teams with over 3000 runs in first innings
    batting_match_winning_df = batting_match_winning_df[(batting_match_winning_df['Team 1 Runs Scored'] >= 300)]

    # Create a new column 'Winning Team' with 1 if 'Team' equals 'Winner', else 0
    batting_match_winning_df['Winning Team'] = np.where(
        batting_match_winning_df['Team 1'] == batting_match_winning_df['Winner'], 1, 0)

    # Map values in 'Winning Team' column to labels
    batting_match_winning_df['Result'] = batting_match_winning_df['Winning Team'].map({0: 'Losing', 1: 'Winning'})

    # plot the pie chart
    get_pie(batting_match_winning_df, 'Result', 'Winning and Losing %age of teams scoring over 300 (1st innings)',
            'over_300')

    batting_300_df = batting_match_winning_df.copy()

    batting_300_df = batting_300_df[['Team 1', 'Winner']]

    # Get total value counts where 'Team 1' is equal to 'Winner'
    total_lost_counts = batting_300_df[batting_300_df['Team 1'] == batting_300_df['Winner']].groupby(
        'Team 1').size().reset_index(name='Total Wins')

    # Display the total value counts
    total_lost_counts = total_lost_counts.sort_values(by='Total Wins', ascending=False)

    # plot the bar chart to show total number of time teams have lost after getting bowled out for less than 300 before 50 overs
    get_bar(total_lost_counts, 'Team 1', 'Total Wins', 'Teams winning after scoring over 300 (1st innings)',
            None, 'Team Name',
            'Win count', 'Teams_over_300_wins')


def run_analysis():
    # get team standings on the points table (group stages)
    df_points = pd.read_csv('data/processed/points_table.csv')
    get_team_standings(df_points)

    # analyze matchwise trend of each team during group stages
    df_match_summary = pd.read_csv('data/processed/match_summary.csv')
    get_teams_matchwise_trend(df_match_summary)

    # get average runs per wicket and average wickets lost per match for all teams (group stages)
    # get correlation among these factors
    get_overall_team_stats(df_match_summary, df_points)

    # get winning and losing trend of teams batting 1st and second
    get_first_second_innings_stats(df_match_summary, df_points)

