#encoding: utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
import os

from sho import make, algo, iters, plot, num, bit, pb

########################################################################
# Interface
########################################################################

if __name__=="__main__":
    import argparse

    # Dimension of the search space.
    d = 2

    can = argparse.ArgumentParser()

    can.add_argument("-n", "--nb-sensors", metavar="NB", default=3, type=int,
            help="Number of sensors")

    can.add_argument("-r", "--sensor-range", metavar="RATIO", default=0.3, type=float,
            help="Sensors' range (as a fraction of domain width, max is √2)")

    can.add_argument("-w", "--domain-width", metavar="NB", default=30, type=int,
            help="Domain width (a number of cells). If you change this you will probably need to update `--target` accordingly")

    can.add_argument("-i", "--iters", metavar="NB", default=100, type=int,
            help="Maximum number of iterations")

    can.add_argument("-s", "--seed", metavar="VAL", default=None, type=int,
            help="Random pseudo-generator seed (none for current epoch)")

    solvers = ["num_greedy","bit_greedy","recuit_simule","genetique","random"]
    can.add_argument("-m", "--solver", metavar="NAME", choices=solvers, default="num_greedy",
            help="Solver to use, among: "+", ".join(solvers))

    can.add_argument("-t", "--target", metavar="VAL", default=30*30, type=float,
            help="Objective function value target")

    can.add_argument("-y", "--steady-delta", metavar="NB", default=50, type=float,
            help="Stop if no improvement after NB iterations")

    can.add_argument("-e", "--steady-epsilon", metavar="DVAL", default=0, type=float,
            help="Stop if the improvement of the objective function value is lesser than DVAL")

    can.add_argument("-a", "--variation-scale", metavar="RATIO", default=0.3, type=float,
            help="Scale of the variation operators (as a ration of the domain width)")


   
    the = can.parse_args()

    NB_RUNS = 10
    #THRESHOLD correspond à la liste des seuils qu'on utilise pour calculer l'EAF
    THRESHOLD = [500,600,650,660,670,675]

    #all_historique contient l'historique de toutes les runs lancés
    all_historique = []

    # Minimum checks.
    assert(0 < the.nb_sensors)
    assert(0 < the.sensor_range <= math.sqrt(2))
    assert(0 < the.domain_width)
    assert(0 < the.iters)

    # Do not forget the seed option,
    # in case you would start "runs" in parallel.
    np.random.seed(the.seed)

    # Weird numpy way to ensure single line print of array.
    np.set_printoptions(linewidth = np.inf)


    history = []



    for run in range(NB_RUNS):

        # Common termination and checkpointing.
        history = []
        # if run == 0:
        iteration = make.iter(
                    iters.several,
                    agains = [
                        make.iter(iters.max,
                            nb_it = the.iters),
                        make.iter(iters.save,
                            filename = the.solver+".csv",
                            fmt = "{it} ; {val} ; {sol}\n"),
                        make.iter(iters.log,
                            fmt="\r{it} {val}"),
                        make.iter(iters.history,
                            history = history),
                        make.iter(iters.target,
                            target = the.target),
                        iters.steady(the.steady_delta, the.steady_epsilon)
                    ]
                )



        # Erase the previous file.
        with open(the.solver+".csv", 'w') as fd:
            fd.write("# {} {}\n".format(the.solver,the.domain_width))

        val,sol,sensors = None,None,None
        if the.solver == "num_greedy":
            val,sol = algo.greedy(
                    make.func(num.cover_sum,
                        domain_width = the.domain_width,
                        sensor_range = the.sensor_range,
                        dim = d * the.nb_sensors),
                    make.init(num.rand,
                        dim = d * the.nb_sensors,
                        scale = the.domain_width),
                    make.neig(num.neighb_square,
                        scale = the.variation_scale,
                        domain_width = the.domain_width),
                    iteration
                )
            sensors = num.to_sensors(sol)

        elif the.solver == "bit_greedy":
            val,sol = algo.greedy(
                    make.func(bit.cover_sum,
                        domain_width = the.domain_width,
                        sensor_range = the.sensor_range,
                        dim = d * the.nb_sensors),
                    make.init(bit.rand,
                        domain_width = the.domain_width,
                        nb_sensors = the.nb_sensors),
                    make.neig(bit.neighb_square,
                        scale = the.variation_scale,
                        domain_width = the.domain_width),
                    iteration
                )
            sensors = bit.to_sensors(sol)


        elif the.solver == "random":
            val,sol = algo.random(
                    make.func(bit.cover_sum,
                        domain_width = the.domain_width,
                        sensor_range = the.sensor_range,
                        dim = d * the.nb_sensors),
                    make.init(bit.rand,
                        domain_width = the.domain_width,
                        nb_sensors = the.nb_sensors),
                    iteration
                )
            sensors = bit.to_sensors(sol)

        elif the.solver == "recuit_simule":
            val,sol = algo.recuit_simule(
                    make.func(bit.cover_sum,
                        domain_width = the.domain_width,
                        sensor_range = the.sensor_range,
                        dim = d * the.nb_sensors),
                    make.init(bit.rand,
                        domain_width = the.domain_width,
                        nb_sensors = the.nb_sensors),
                    make.neig(bit.neighb_square,
                        scale = the.variation_scale,
                        domain_width = the.domain_width),
                    iteration
                )
            sensors = bit.to_sensors(sol)
        
       



        elif the.solver == "genetique":
            NUM_CROSSOVER = 0
            val,sol = algo.genetique(
                    make.func(bit.cover_sum,
                        domain_width = the.domain_width,
                        sensor_range = the.sensor_range,
                        dim = d * the.nb_sensors),
                    make.init(bit.rand,
                        domain_width = the.domain_width,
                        nb_sensors = the.nb_sensors),
                    make.neig(bit.neighb_square,
                        scale = the.variation_scale,
                        domain_width = the.domain_width),
                    100,
                    iteration,
                    NUM_CROSSOVER
                )
            sensors = bit.to_sensors(sol)

        
        temp = []
        for i in range(len(history)):
            temp.append(history[i][0])
        all_historique.append(temp)
            


    #Affichage des graphiques

    for thres in THRESHOLD:
        threshold_atteint = [0]*the.iters
        for histo in all_historique:
            for i in range(len(histo)):
                if histo[i] > thres:
                    threshold_atteint[i] += 1/NB_RUNS
        
        plt.plot(range(the.iters),threshold_atteint)
        plt.title('EAF pour un seuil de ' + str(thres) + " avec un algorithme : " + str(the.solver))
        plt.xlabel("Nombre d'appels à la fonction objectif")
        plt.ylabel('Probabilité de dépasser le seuil')
        plt.show()
        


