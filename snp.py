#encoding: utf-8
import math
import numpy as np
import matplotlib.pyplot as plt

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


    # Common termination and checkpointing.
    history = []
    iters = make.iter(
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
                iters
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
                iters
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
                iters
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
                iters
            )
        sensors = bit.to_sensors(sol)
    


    elif the.solver == "genetique":
        NUM_CROSSOVER = 0
        POP_SIZE = 100
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
                POP_SIZE,
                iters,
                NUM_CROSSOVER
            )
        sensors = bit.to_sensors(sol)

    

    # Fancy output.
    print("\n{} : {}".format(val,sensors))

    shape=(the.domain_width, the.domain_width)

    fig = plt.figure()

    if the.nb_sensors ==1 and the.domain_width <= 50:
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        f = make.func(num.cover_sum,
                        domain_width = the.domain_width,
                        sensor_range = the.sensor_range * the.domain_width)
        plot.surface(ax1, shape, f)
        plot.path(ax1, shape, history)
    else:
        ax2=fig.add_subplot(111)



    domain = np.zeros(shape)
    domain = pb.coverage(domain, sensors,
            the.sensor_range * the.domain_width)
    domain = plot.highlight_sensors(domain, sensors)
    ax2.imshow(domain)

    plt.show()

