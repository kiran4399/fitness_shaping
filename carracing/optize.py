
from pulp import *
import numpy as np
import padnas as pd
import re
import matplotlib.pyplot as plt
from IPython.display import Image


# Download the dataset from https://www.kaggle.com/rounakbanik/ted-talks

# Read the dataset into pandas dataframe, convert duration from seconds to minutes
ted = pd.read_csv('ted_main.csv', encoding='ISO-8859-1')
ted['duration'] = ted['duration'] / 60
ted = ted.round({'duration': 1})

# Select subset of columns & rows (if required)
# data = ted.sample(n=1000) # 'n' can be changed as required
data = ted
selected_cols = ['name', 'event', 'duration', 'views']
data.reset_index(inplace=True)
data.head()

# create LP object,
# set up as a maximization problem --> since we want to maximize the number of TED talks to watch
prob = pulp.LpProblem('WatchingTEDTalks', pulp.LpMaximize)



# create decision - yes or no to watch the talk?
decision_variables = []
for rownum, row in data.iterrows():
    # variable = set('x' + str(rownum))
    variable = str('x' + str(row['index']))
    variable = pulp.LpVariable(str(variable), lowBound = 0, upBound = 1, cat = 'Integer') # make variable binary
    decision_variables.append(variable)
    
print('Total number of decision variables: ' + str(len(decision_variables)))

# Create optimization Function
total_views = ''
for i,k in enumerate(decision_variables):
    for j in range(len(w[i])):
    	formula = k*w[i,j]
    	total_views += formula
    total_views += 
	total_views -= 
	prob += total_views
# print('Optimization function: ' + str(total_views))

# Contraints
total_time_available_for_talks = 10*60 # Total time available is 10 hours . Converted to minutes
total_talks_can_watch = 25 # Don't want an overload information


# Create Constraint 1 - Time for talks
total_time_talks = ''
for rownum, row in data.iterrows():
    for i,  talk in enumerate(decision_variables):
        if rownum == i:
            formula = row['duration']*talk
            total_time_talks += formula
            
prob += (total_time_talks == total_time_available_for_talks)



# Create Constraint 2 - Number of talks
total_talks = ''

for rownum, row in data.iterrows():
    for i, talk in enumerate(decision_variables):
        if rownum == i:
            formula = talk
            total_talks += formula
            
prob += (total_talks == total_talks_can_watch)

print(prob)
prob.writeLP('WatchingTEDTalks.lp')

optimization_result = prob.solve()

assert optimization_result == pulp.LpStatusOptimal
print('Status:', LpStatus[prob.status])
print('Optimal Solution to the problem: ', value(prob.objective))
print('Individual decision variables: ')
for v in prob.variables():
    print(v.name, '=', v.varValue)