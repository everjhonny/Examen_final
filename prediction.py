# -*- coding:utf-8 -*-
import pandas as pd
import math
import csv
import random
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#Cuando cada equipo no tiene calificación de elo, dale una calificación de elo básica
base_elo = 1600
team_elos = {}
team_stats = {}
X = []
y = []
folder = 'data'

# Calcula el valor elo de cada equipo



    # Calcula el valor elo de cada equipo
def calc_elo(ganador, perdedor):
    k=30
    winner_rank = get_elo(ganador)
    loser_rank = get_elo(perdedor)
    df_ganador = diferencia_goles(ganador)
    w = est_juego(ganador)
    rank_diff = winner_rank - loser_rank
    exp = (rank_diff  * -1) / 400
    we = 1 / (1 + math.pow(10, exp))    
    new_winner_rank = round(winner_rank + (k * df_ganador*(w-we)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff
    return new_winner_rank, new_loser_rank


def est_juego(equipo):
    data1 = pd.read_csv(folder + '/est_temp_2020.csv')
    for index, row in data1.iterrows():
        if row['equipo']==equipo:
            res=1.0*(row['pg'])+(row['pe'])*0.5
    return res
def diferencia_goles(equipo):
    data1 = pd.read_csv(folder + '/est_temp_2020.csv')
    for index, row in data1.iterrows():
        if row['equipo']==equipo:
            dg=row['dg']
            if dg==0 or dg==1:
                dg=1.0  
            elif dg<0:
                dg=0.0
            else:
                dg=(11+dg)/8
                
    return dg
# inicializa el archivo csv de estadísticas del equipo
def initialize_data(Mstat, Ostat, Tstat):
    new_Mstat = Mstat.drop(['puntos_elo', 'sede'], axis=1)#ya esta
    new_Ostat = Ostat.drop(['pos', 'pjv', 'puntos_visitante'], axis=1)
    new_Tstat = Tstat.drop(['pos', 'pjl', 'puntos_local'], axis=1)

    team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='equipo')
    team_stats1 = pd.merge(team_stats1, new_Tstat, how='left', on='equipo')

    return team_stats1.set_index('equipo', inplace=False, drop=True)
# elo 
def get_elo(equipo):
    rk = pd.read_csv(folder + '/ranking.csv')
    for index, row in rk.iterrows():
        if row['equipo']==equipo:
            elo_act=row['puntos_elo']
    return elo_act

def  build_dataSet(all_data):
    print("armando conjunto de datos..")
    X = []
    
    for index, row in all_data.iterrows():
        
        if row['resultado'] == 'vl':#victoria de local
            ganador = row['local']
            perdedor = row['visitante']            
        if row['resultado'] == 'vv':#victoria de visitante
            ganador = row['visitante']
            perdedor = row['local']
        if row['resultado'] == 'e':#empate
            ganador = row['visitante']
            perdedor = row['local']   
        #Obtener el valor de elo inicial de cada equipo
        team1_elo = get_elo(ganador)
        team2_elo = get_elo(perdedor)

        #elo de +100 al equipo en el juego en casa
        if row['resultado'] == 'vl':
            team1_elo += 100
        
        # el ptje elo sera el primer valor de característica para evaluar a cada equipo
        team1_features = [team1_elo]
        team2_features = [team2_elo]

        #estadísticas de cada equipo 
        for key, value in team_stats.loc[ganador].iteritems():
            team1_features.append(value)
        for key, value in team_stats.loc[perdedor].iteritems():
            team2_features.append(value)

       # Asignar aleatoriamente los valores propios de los dos equipos a los lados izquierdo y derecho de los datos para cada juego
       # Y asigne el 0/1 correspondiente al valor y
        if random.random() > 0.5:
            X.append(team1_features + team2_features)
            y.append(0)
        else:
            X.append(team2_features + team1_features)
            y.append(1)

        
        # Actualiza el valor de elo del equipo en función de los datos de este juego.
        new_winner_rank, new_loser_rank = calc_elo(ganador, perdedor)
        team_elos[ganador] = new_winner_rank
        team_elos[perdedor] = new_loser_rank

    return np.nan_to_num(X), np.array(y)

def predict_winner(team_1, team_2, model):
    features = []

    features.append(get_elo(team_1) + 100)
    for key, value in team_stats.loc[team_2].iteritems():
        features.append(value)
        # equipo 1, equipo visitante
    features.append(get_elo(team_2))
    for key, value in team_stats.loc[team_1].iteritems():
        features.append(value)

    # equipo 2, el equipo local
    

    features = np.nan_to_num(features)
    return model.predict_proba([features])

if __name__ == '__main__':

    Mstat = pd.read_csv(folder + '/ranking.csv')
    Ostat = pd.read_csv(folder + '/est_temp_2020.csv')
    Tstat = pd.read_csv(folder + '/est_temp_2020.csv')

    team_stats = initialize_data(Mstat, Ostat, Tstat)

    result_data = pd.read_csv(folder + '/resultados_temp_2020.csv')
    
    X, y = build_dataSet(result_data)

    # Entrena el modelo de red
    print("probando en %d muestras de juego.." % len(X))

   
    model = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000)
    model.fit(X, y)
        
    # validación cruzada de 10 veces para calcular la precisión del entrenamiento
    print("armando validacion cruzada...")
    print(cross_val_score(model, X, y, cv = 26, scoring='accuracy', n_jobs=-1).mean())
    print("matriz",confusion_matrix(y,))
    
    #Utilizando el modelo entrenado para hacer predicciones
    print('Predecir resultados de los nuevos cruces..')
    nuevos_cruces = pd.read_csv(folder + '/cruces.csv')
    result = []
    for index, row in nuevos_cruces.iterrows():
        team1 = row['local']
        team2 = row['visitante']
        pred = predict_winner(team1, team2, model)
        prob = pred[0][0]
      
        if prob==0.5:
            winner = team1
            loser = team2
            tv='em'
            result.append([winner, loser, prob,tv])
            
        if prob > 0.5:
            winner = team1
            loser = team2
            tv='vl'
            result.append([winner, loser, prob,tv])
        else:
            winner = team2
            loser = team1
            tv='vv'
            result.append([winner, loser, 1 - prob,tv])

    with open('Result.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows(['g', 'p', 'pb','resul'])        
        writer.writerows(result)

print(result)








