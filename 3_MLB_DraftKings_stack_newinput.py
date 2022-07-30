import pandas as pd
from pulp import *
import numpy as np


# User inputs - Please make your inputs here
projections_file_path = r'/home/krishnamoorthy/Desktop/Python/python_money/DFS/MLB DK Projections.csv'
ownership_file_path = r'/home/krishnamoorthy/Desktop/Python/python_money/DFS/MLB DK Ownership.csv'
output_file_path = r'/home/krishnamoorthy/Desktop/Python/python_money/DFS/Lineup_Final_stacked.csv'

ownership_cap = 100
salary_cap = 50000
salary_floor = 0
number_of_lineups = 30

slate_needed = 'Main'

# Enter stacking requirements here
stack_count_1 = 4
team_1 = 'SF'
stack_count_2 = 3
team_2 = 'NYY'
stack_count_3 = 0
team_3 = 'WAS'

# Enter must haves and must not haves here
Must_haves = []
Must_not_haves = []

df1 = pd.read_csv(projections_file_path,
                  usecols=['Name', 'Fpts', 'Position', 'Team', 'Salary', 'Slate'])
df2 = pd.read_csv(ownership_file_path,
                  usecols=['Name', 'Salary', 'Position', 'Team', 'Ownership %'])

result = df1.merge(df2, how='outer').dropna(subset=['Fpts'])
result['Ownership %'] = result['Ownership %'].fillna(0)

result.replace(['1B/2B', '1B/3B', '1B/OF', '2B/3B', '2B/OF', '2B/SS', '3B/OF',
                'C/OF', 'UTIL', '1B/SS', '3B/SS', 'C/1B', 'SS/OF'], ['1B', '1B', '1B', '2B', '2B', '2B', '3B', 'C', '2B', '1B', '3B', 'C', 'SS'], inplace=True)

result.rename(columns={'Fpts': 'fpts', 'Position': 'DK Position',
              'Ownership %': 'proj_own', 'Salary': 'Slate Salary'}, inplace=True)

result['Slate Salary'] = result['Slate Salary'].astype(
    'str').str.replace(',', '').astype(int)


availables = result.groupby(
    ["DK Position", "Name", "fpts", "Slate Salary", 'Team', 'proj_own', 'Slate']).agg('count')
availables = availables.reset_index()


# We're putting each player into one position only; those that have felxibility like '1B/2B' should be put in either 1B or 2B to avoid error
availables.replace(['1B/2B', '1B/3B', '1B/OF', '2B/3B', '2B/OF', '2B/SS', '3B/OF',
                   'C/OF'], ['1B', '1B', '1B', '2B', '2B', '2B', '3B', 'C'], inplace=True)


availables = availables[~availables.Name.isin(Must_not_haves)]

salaries = {}
points = {}
ownership = {}
teams = availables.Team.unique()

lineups_dict = {}

availables = availables[availables['Slate Salary'] > salary_floor]


# We're describing the number of positions here

pos_num_available = {
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "OF": 3,
    "SS": 1,
    "P": 2,
    "C": 1
}

mean_AvgPointsPerGame = 0

availables['stick1'] = 0
availables['stick2'] = 0
availables['stick3'] = 0
availables.loc[availables.Team == team_1, 'stick1'] = 1
availables.loc[availables.Team == team_2, 'stick2'] = 1
availables.loc[availables.Team == team_3, 'stick3'] = 1

availables['studs'] = 0
availables.loc[availables.Name.isin(Must_haves), 'studs'] = 1

# availables
n = len(Must_haves)
# availables.head(15)

availables = availables[availables.Slate == slate_needed]

for j in range(number_of_lineups+1):

    prob = LpProblem('DFS_MLB', LpMaximize)
    player_vars = [LpVariable(f'player_{row.Name}', cat='Binary')
                   for row in availables.itertuples()]

    # total salary constraint
    prob += lpSum(availables['Slate Salary'].iloc[i] * player_vars[i]
                  for i in range(len(availables))) <= salary_cap

    # create a helper function to return the number of players assigned each position
    def get_position_sum(player_vars, df, position):
        return lpSum([player_vars[i] * (position in df['DK Position'].iloc[i]) for i in range(len(df))])

    for pos in pos_num_available.keys():
        prob += get_position_sum(player_vars, availables,
                                 pos) == pos_num_available[pos]

    prob += lpSum(player_vars[i] for i in range(len(availables))) == 10

    prob += lpSum([availables.stick1.iloc[i] * player_vars[i]
                  for i in range(len(availables))]) == stack_count_1

    prob += lpSum([availables.stick2.iloc[i] * player_vars[i]
                  for i in range(len(availables))]) == stack_count_2

    prob += lpSum([availables.stick3.iloc[i] * player_vars[i]
                  for i in range(len(availables))]) == stack_count_3

    prob += lpSum([availables.studs.iloc[i] * player_vars[i]
                  for i in range(len(availables))]) == n

    prob += lpSum([availables.fpts.iloc[i] * player_vars[i]
                  for i in range(len(availables))])

    prob += lpSum(availables['proj_own'].iloc[i] * player_vars[i]
                  for i in range(len(availables))) <= ownership_cap

    # solve and print the status

    if(j != 0):
        prob += lpSum([availables.fpts.iloc[i] * player_vars[i]
                      for i in range(len(availables))]) <= mean_AvgPointsPerGame-0.1

    prob.solve()
    if(LpStatus[prob.status] != 'Optimal'):
        ownership_cap = ownership_cap + 50
        continue
    # if(max([sum([player_vars[jj].value() for jj in range(len(availables)) if availables.Team.iloc[jj] == i]) for i in availables.Team.unique()]) < stack_count):
    #     j = j-1
    #     continue

    total_salary_used = 0
    mean_AvgPointsPerGame = 0
    total_ownership = 0
    lineup = []
    for i in range(len(availables)):
        if player_vars[i].value() == 1:
            row = availables.iloc[i]
            lineup.append('_'.join(row.Name.split(' ')) + '_' + row.Team)
#             print(row['DK Position'], row.Name, row.Team, row['Slate Salary'], row.fpts)
            total_salary_used += row['Slate Salary']
            total_ownership += row['proj_own']
            mean_AvgPointsPerGame += row.fpts
#     mean_AvgPointsPerGame /= 9  # divide by total players in roster to get a mean
#     print(mean_AvgPointsPerGame)
    lineup.append(mean_AvgPointsPerGame)
    lineup.append(total_ownership)
    lineups_dict[j] = lineup

df = pd.DataFrame(lineups_dict, index=[
                  '1B', '2B', '3B', 'C', 'OF1', 'OF2', 'OF3', 'P1', 'P2', 'SS', 'TotalPoints', 'TotalOwnership']).T

#   Saving in your output file
df.to_csv(output_file_path)
# print(result['Slate Salary'].dtype)
