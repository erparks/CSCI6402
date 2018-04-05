import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

###################################################
# Split the given dataframe into an array of win share data and
#  a matrix holdig the input data
###################################################
def InOutSplit(df):
    training_output = df[['WS']]
    #Use all given data other than the values dropped
    training_input = df.drop(['WS', 'OWS', 'DWS', 'Player', 'Tm', 'Pos', 'blanl', 'blank2'], axis=1, inplace=False)

    #Use only the specified values
    #training_input = df[['TRB', '3P%', '3P', '2P%', '2P', 'AST', 'STL']]

    return training_input, training_output

###################################################
# Split the given dataframe into a training set and a test set.
#  Returns a 90/10 split respectively.
###################################################
def TrainTestSplit(df):    
    remove_n = int(len(df) * 0.1)
    drop_indices = np.random.choice(len(df), remove_n, replace=False)
    train_split = df.drop(drop_indices)
    test_split = df.iloc[drop_indices]

    return train_split, test_split

###################################################
# Create and train a model with the given input/output data
###################################################
def Train(input_mat, output_mat):
    #update the values in the () after hidden_layer_sizes to change the structure of the neural net
    #for example: hidden_layer_sizes=(30,30,30) creates a 3 layer network with 30 nodes each
    reg = MLPRegressor(hidden_layer_sizes=(10,10,10), solver='adam', max_iter=700)
    reg.fit(input_mat, output_mat)
    
    return reg

#Read in the .csv
df = pd.read_csv('Seasons_Stats.csv', index_col=0)

#print("===== " + str(year_range) + " - " + str(year_range + 10) + " =====")

#Select the appropriate data
era = pd.DataFrame(columns=df.columns)
#Remove data from years below the lower limit
cond = df.Year >= 1973
rows = df.loc[cond, :]
#Remove data from years after the top limit
cond = df.Year < 1979
rows = rows.loc[cond, :]

#Create dataframe to hold selected data
era = era.append(rows, ignore_index=True)
#Replace missing data with 0
era.fillna(0, inplace=True)

#Split the data into training and testing sets
era_train, era_test = TrainTestSplit(era)

#Split the sets into net inputs and net outputs
training_input, training_output = InOutSplit(era_train)
test_input, test_output = InOutSplit(era_test)

#Train and test the model 10 times, recording the error for each iteration
errors = []
for i in range(10):
    model = Train(training_input.as_matrix(), training_output.values.ravel())
    error = np.absolute(np.subtract(model.predict(test_input), test_output.values.ravel()))
    errors.append(np.average(error))
    print('avg error for iteration '+ str(i) + ": " + str(np.average(error)))

#Print average error
print('Win share standard deviation: ' + str(np.std(era['WS'])))
print(str(sum(errors) / float(len(errors))))


