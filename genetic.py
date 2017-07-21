# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:20:44 2017

@author: shrr001
"""


from random import randint
import random
from isolation import Board
from game_agent import AlphaBetaPlayer, MinimaxPlayer, custom_score_2
import numpy as np
import heapq
import progressbar
from scipy import stats

def individual(weight):
    # creates an individual 
    return AlphaBetaPlayer(score_fn=custom_score_2, weight=weight)

def population(count):
    # creates a list of individuals 
    return [ individual(random.uniform(0,1)) for x in range(count) ]



def evolve(pop, gamesFactor=2, retain=0.2, random_select=0.05, mutate=0.01):
    # Determine the parents to breed from the population
    agent_score = {}
    numGames = len(pop) * gamesFactor
    bar = progressbar.ProgressBar()

    for game in bar(range(numGames)):
        competitors = random.sample(pop, 2)
        game = Board(competitors[0], competitors[1])
        winner, history, outcome = game.play()
        competitors.remove(winner)
        loser = competitors[0]
        if winner not in agent_score.keys():
            agent_score[winner] = 1
        else:
            agent_score[winner] += 1
        if loser not in agent_score.keys():
            agent_score[winner] = -1
        else:
            agent_score[loser] -= 1        
        
    top_performers_size = int(retain * len(pop))
    bottom_performers_size = len(pop) - top_performers_size
    rand_select_size = int(len(pop) * random_select)
    top_perfomers = heapq.nlargest(top_performers_size, agent_score, key=agent_score.get)
    bottom_performers = heapq.nsmallest(bottom_performers_size, agent_score, key=agent_score.get)
    parents = top_perfomers + random.sample(bottom_performers, rand_select_size)
    random.shuffle(parents)
    

    # Create children
    numChildren = len(pop) - len(parents)
    
    children = []
    for i in range(numChildren):
        par = random.sample(parents, 2)
        father = par[0]
        mother = par[1] 
        if father.w > mother.w:
            childWeight = random.randrange(father.w*0.95, mother.w*1.05)
        else:
            childWeight = random.randrange(mother.w*0.95, father.w*1.05)
        child = individual(childWeight)
        children.append(child)
        
    new_pop = parents + children

    mutated_pop = []
    # Randomly mutate some of the new population
    for agent in new_pop:
        if mutate > random.uniform(0,1):
            print('Mutate')
            newWeight = agent.w + random.uniform(-0.05,0.05)
            if newWeight < 0:
                newWeight = 0
            if newWeight > 1:
                newWeight = 1
            mutated_pop.append(individual(newWeight))
        else:
            mutated_pop.append(agent)
    return mutated_pop
            
            
        
    
if __name__ == "__main__":
    pop_count = 30
    evolution_cyles = 20
    pop = population(pop_count)
    history = []
    for i in range(evolution_cyles):
        print(i)
        pop = evolve(pop, gamesFactor=5, retain=0.2, random_select=0.05, mutate=0.03)
        best_weights = [i.w for i in pop]
        print(stats.describe(best_weights))
        history.append(best_weights)
      
    print('Evolultion Results:')
    [stats.describe(x) for x in history]

    
