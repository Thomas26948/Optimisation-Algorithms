########################################################################
# Algorithms
########################################################################
import numpy as np


def random(func, init, again):
    """Iterative random search template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 0
    while again(i, best_val, best_sol):
        sol = init()
        val = func(sol)
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol


def greedy(func, init, neighb, again):
    """Iterative randomized greedy heuristic template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 1
    while again(i, best_val, best_sol):
        sol = neighb(best_sol)
        val = func(sol)
        # Use >= and not >, so as to avoid random walk on plateus.
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol

# TODO add a simulated-annealing template.
def recuit_simule(func,init,neighb,again):
    T = 1.2
    # T = 1000
    true_sol = init()
    true_val = func(true_sol)
    best_sol = true_sol
    best_val = true_val
    i = 1
    while again(i,true_val,true_sol):
        sol = neighb(best_sol)
        val = func(sol)
        if val >= best_val:
            best_val = val
            best_sol = sol
            true_sol = best_sol
            true_val = best_val
        elif np.exp(-(best_val - val)/T) >= np.random.random():
            best_val = val
            best_sol = sol
        T = 0.99 * T   
        i += 1 
    return true_val,true_sol

import random

# TODO add a population-based stochastic heuristic template.
def genetique(func,init,neighb,pop_size,again,num_crossover=0):

    #Le crossover sélection aléatoirement l'une des deux populations parentes
    #On prend ensuite le voisinage de cette solution
    def crossover(pop_A,pop_B):
        if np.random.random() > 0.5:
            pop = pop_A
        else:
            pop = pop_B
        #Phase de mutation avec neighb
        sol = neighb(pop)
        val = func(sol)
        return [val,sol]

    #Avec le crossover2, on obtient une population qui est une combinaison linéaire des 2 populations parentes
    #Le coefficient de la combinaison linéaire est choisi aléatoirement 
    def crossover2(pop_A,pop_B):
        Xa,Ya = np.where(pop_A == 1)
        Xb,Yb = np.where(pop_B == 1)
        L = [( int((np.random.random())*(xa-xb)+xb ),int((np.random.random())*(ya-yb)+yb)) for xa,ya,xb,yb in zip(Xa,Ya,Xb,Yb)]
        pop = np.zeros(pop_A.shape)
        for x,y in L:
            pop[x][y] = 1
        #Phase de mutation avec neighb
        sol = neighb(pop)
        val = func(sol)
        return [val,sol]

    #Choix de la fonction de crossover
    f_crossover = crossover if num_crossover == 0 else crossover2 
    
    pop = []
    n = 1
    
    #Génération d'une population initiale
    for i in range(pop_size):
        sol = init()
        pop.append([func(sol),sol])


    #On trie notre population de manière décroissante, de sorte que le meilleur score soit à l'indice 0.
    pop.sort( key=lambda x: x[0])
    pop = pop[::-1]
    best_val, best_sol = pop[0]

    while again(n,best_val,best_sol):
        temp_pop = []

        #Phase de sélection et de variation
        # la nouvelle population est constitué à 80 % des 20% anciennes meilleurs pop, et à 20% des 80% moins bonnes pop
        for i in range(int(0.8*pop_size)):
            temp_pop.append(f_crossover(pop[np.random.randint( 0.2 * pop_size ) ][1],pop[np.random.randint( 0.2 * pop_size ) ][1]))

        for i in range( int( 0.2 * pop_size )):
            temp_pop.append(f_crossover(pop[np.random.randint( 0.2 * pop_size,pop_size ) ][1],pop[np.random.randint( 0.2 * pop_size ,pop_size) ][1]))

        temp_pop.sort( key=lambda x: x[0])
        temp_pop = temp_pop[::-1]
        #Phase de remplaçage
        pop = temp_pop

        if best_val < pop[0][0]:
            best_val = pop[0][0]
            best_sol = pop[0][1]

        n += 1 
        
    return best_val,best_sol



