import json
from random import shuffle
from sklearn.neural_network import MLPClassifier
import numpy as np


######################################################
# Creates a dictionary with keys for each zone - area
#  combination.
######################################################
def create_shot_dict(areas, dists):
        shot_dict = {}
        for area in areas:
                for dist in dists:
                        shot_dict[area+dist] = 0

        return shot_dict

####################################################
# Populate a dictionary that maps gameIDs to
#  another dictionary which maps zones on the court
#  to number of shots within that game
####################################################
def load_data(filename):
        data = json.load(open(filename))

        game_dict = {}
        for shot in data['resultSets'][0]['rowSet']:
                if not str(shot[1]) in game_dict:
                        game_dict[str(shot[1])] = create_shot_dict(shot_areas, shot_distances)

                game_dict[str(shot[1])][shot[14]+shot[15]] += 1

        return game_dict

####################################################
# Pull an array of shot frequencies out
#  of the given dictionary for each game
####################################################
def extract_shots(game_set):
        
        all_shots = []
        for game in game_set:
                shots = []
                for shot in game_set[game]:
                        shots.append(game_set[game][shot])
                all_shots.append(shots)

        return all_shots

####################################################
# Prints the average number of shots from each bin
#  in the given game_set
####################################################
def avg_game_calc(game_set):
        avg_game = [0] * 30

        for game in np.array(game_set):
                avg_game += game
               
                
        avg_game = avg_game/float(len(game_set))
        for zone, percent in zip(zones, avg_game):
                print zone + "\t" + str(percent)


#Zones and Distances on the court -- Constant through all data sources
shot_areas = [u'Center(C)', u'Left Side(L)', u'Right Side Center(RC)', u'Right Side(R)', u'Back Court(BC)', u'Left Side Center(LC)']
shot_distances = [u'8-16 ft.', u'Back Court Shot', u'16-24 ft.', u'24+ ft.', u'Less Than 8 ft.']
zones = create_shot_dict(shot_areas, shot_distances).keys()
        
if __name__ == "__main__":
        
        #load shot data
        wins_dict   = load_data('AllLebronShotsWins.json')
        losses_dict = load_data('AllLebronShotslosses.json')

        ##wins_dict   = load_data('AllDurantShotsWins.json')
        ##losses_dict = load_data('AllDurantShotslosses.json')

        ##wins_dict   = load_data('AllGiannisShotsWins.json')
        ##losses_dict = load_data('AllGiannisShotslosses.json')

        ##wins_dict   = load_data('AllGordonShotsWins.json')
        ##losses_dict = load_data('AllGordonShotslosses.json')

        ##wins_dict   = load_data('AllHardenShotsWins.json')
        ##losses_dict = load_data('AllHardenShotslosses.json')

        ##wins_dict   = load_data('AllCurryShotsWins.json')
        ##losses_dict = load_data('AllCurryShotslosses.json')

        #extract and format data
        win_shots  = extract_shots(wins_dict)
        loss_shots = extract_shots(losses_dict)

        print'========== avergage actual win =========='
        avg_game_calc(win_shots)

        print'========== avergage actual loss =========='
        avg_game_calc(loss_shots) 

        #used to track accuracy accross the 20 loops below
        acc = []

        #Empty array to track average win or loss across the 20 loops below
        avg_games_total = [0] * 30

        #Number of games that ended up being wins or losses
        total_games = 0.

        for i in range(20):
                shuffle(win_shots)

                win_shots = win_shots[:len(loss_shots)]

                #Create correspinding output data where 1 = win and 0 = loss
                win_output  = [1] * len(win_shots)
                loss_output = [0] * len(loss_shots)

                #combine the wins and losses and then shuffle the order
                all_games   = win_shots + loss_shots
                all_outputs = win_output + loss_output
                x = list(zip(all_games, all_outputs))
                shuffle(x)

                #extract the input and outputs from the shuffled list
                eq_game, eq_output = zip(*x)
                eq_game = np.asarray(eq_game)
                eq_output = np.asarray(eq_output)


                #split into test and training splits
                train_games = eq_game[:int(len(eq_game)*0.9)]
                test_games  = eq_game[int(len(eq_game)*0.9):]

                train_output = eq_output[:int(len(eq_game)*0.9)]
                test_output  = eq_output[int(len(eq_game)*0.9):]

                #initialize
                clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(60,30,20), verbose=0, max_iter=10000)

                #train
                clf.fit(train_games, train_output)

                #predict
                test_pred = clf.predict(test_games)

                #Calculate accuracy of predictions
                corr = 0.
                win_pred = 0
                for o, p in zip(test_output, test_pred):
                                                if(p == o):
                                                        corr += 1
                                                if(p == 1):
                                                        win_pred += 1

                acc.append(corr/len(test_output))

                #see average predicted win or loss
                
                selected_games = []
                avg_game = []
                for game, p in zip(test_games, test_pred):
                        #change to p == 0 for losses
                        if(p == 1):
                                selected_games.append(game)

                for game in selected_games:
                        if (avg_game == []):
                                avg_game = game
                        else:
                                for i in range(0, len(game)-1):
                                        avg_game[i] += game[i]

                if(win_pred > 0):
                        avg_games_total += avg_game
                        total_games += len(selected_games)


        #Print average accuract across all tests
        print(str(sum(acc)/float(len(acc))))


        #Print average shot frequencies for the collected games (wins or losses)
        avg_games_total = avg_games_total/total_games

        for zone, percent in zip(zones, avg_games_total):
                print zone + "\t" + str(percent)

        












