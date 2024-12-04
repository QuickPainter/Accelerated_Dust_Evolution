import numpy as np
import math
import time
import random

# Constants
#  number of representative particles 
# ntot = 200             
#  mass of a all the particles captured by a specific representative particle
representative_particle_mass = 10e20           
# Initial mass of all particles
m0 = 1.0                 
# Simulation end time
tend = 15.0             
# Write to file at certain time step set by dtout
dtout = 3.0              
# Bins for mass histogram
nbins = 200             

# depropensity_sume a particle class following the representative particle scheme, i.e. it has a set mass of all its particles and the number of particles its keeping track of
class RP:
    def __init__(self, num_par, mass):
        self.num_par = num_par  # Number of physical particles represented
        self.mass = mass  # Mass of the representative particle

def main(tp, run):

    #  number of representative particles 
    ntot = tp             

    print('Simulating run number: ', run)

    # Initialize a list containing each class of representative particles, so we have a 'representative_particles'  of representative particles
    ## all representative particles are the same to start, they all model particles with mass m0 

    representative_particles = []
    for i in range(ntot):
        # as all our particles start with mass unity, the number of particles is the mass of the group of representative particles dividide by the starting mass (in our case we set =1)
        number_of_particles = representative_particle_mass / m0
        # can then add this to our list of representative particles
        representative_particles.append(RP(number_of_particles, m0))

    # calculate as volume, since V = Mtot / p
    vol = ntot * representative_particle_mass  / m0

    ##############################
    ### SIMULATION SETTINGS ######
    sim_time = 0.0             # Current simulation time
    dtime = 0.0                # time till next reaction
    tout = dtout               # Next time of writing to disk
    ### SIMULATION SETTINGS ######
    ##############################

    ##############################
    ### R-LEAPING SETTINGS ######
    num_reactions_leap = 500            # number of reactions desired for R leaping
    ### R-LEAPING SETTINGS ######
    ##############################

    # Collision rate matrix --> this will determine the collisions that happen
    ## initialize as a matrix of zeroes, with an element for each representative particle interaction
    collision_rates_matrix = np.zeros((ntot, ntot))

    # this is the collision rate probability of particle i, which you get by summing up all the collision rate probabilities with other particles it could collide with
    # we will leave it as zero to start
    particle_collision_rates = np.zeros(ntot)

    # Histogram bins that we will track the evolution of particles masses in
    mass_histogram = np.zeros(nbins + 1)

    # this is the mass densities corresponding to those bins
    mass_densities = np.zeros(nbins)

    # we'll initialize the actual histogram bins on a log scale because it spans many orders of magnitude
    # the first bin will be the size of our starting particle, because we won't have any particles smaller than that (we're only modeling collision + growth, not fragmentation)
    mass_histogram[0] = m0

    # Our total mass is equal to the representative_particle_mass * the number of representative particles we have. In theory, if all the particles collided we could grow this big (as we are modelling collision as just the sum of the two particles)
    # we first propensity_sumd the number of orders of magnitude this total mass range spans 
    nord = math.log10(representative_particle_mass * ntot / m0)

    # then we propensity_sumd the division of this range of magnitudes between our mass bins.
    # i.e., what jump in mass does each bin in our histogram correspond to?
    ll = nord / nbins

    # label each mass bin in our histogram
    for i in range(1, nbins + 1):
        mass_histogram[i] = m0 * 10 ** (ll * i)

    # now we need to actually calculate the collision rates of each particle. We calculate the collision rates following A. Zsom and C. P. Dullemond (2008), using a linear coagulation kernel (just the sum of the masses), and ignoring the velocities of the particles
    for i in range(ntot):
        for j in range(ntot):
            collision_rates_matrix[i][j] = representative_particles[i].num_par * 0.5 * (representative_particles[i].mass + representative_particles[j].mass) / vol

    # for each particle, we sum the collision rates for that particle with all of its potential particle colliders, to get the overall collision rate of that particle with any other particle 
    for i in range(ntot):
        particle_collision_rates[i] = np.sum(collision_rates_matrix[i])


    # keep track of time
    start_time = time.time()
    tot_num_collisions = 0
    # Now we begin our simulation
    while sim_time < tend:

        # Total collision rate until some collision happens
        totrate = np.sum(particle_collision_rates)

        # sample time from gamma 
        dtime = np.random.gamma(num_reactions_leap, 1/totrate)

        # print('dtime', dtime)

        # keep track of representative particles that have been chosen to collide
        representative_colliding_particle1_list = []


        ## assign number of collisions to different representative particles

        for reaction in range(num_reactions_leap):
            tot_num_collisions+=1

            # Select first particle for collision, multiply uniform value by the total rate
            random_value = random.random() * totrate

            propensity_sum = particle_collision_rates[0]
            colliding_particle1 = 0

            # iterate through until random number is larger than the sum of the propensities up until that point
            while random_value > propensity_sum and colliding_particle1 < ntot - 1:
                colliding_particle1 += 1
                propensity_sum += particle_collision_rates[colliding_particle1] 

            # keep track of which particle was chosen
            representative_colliding_particle1_list.append(colliding_particle1)

            # choose second particle for collision, by looking at the particle collision rates of the first particle you chose
            random_value = random.random() * particle_collision_rates[colliding_particle1]

            propensity_sum = collision_rates_matrix[colliding_particle1][0]
            colliding_particle2 = 0

            while random_value > propensity_sum and colliding_particle2 < ntot - 1:
                colliding_particle2 += 1
                propensity_sum += collision_rates_matrix[colliding_particle1][colliding_particle2]

            # perform collision --> representative particle list gets updated, so that the representative particle corresponding to the first colliding particle now models a group of particle whose new mass is set by the addition of the second colliding particle. 
            # we don't need to worry about updating the second colliding particle --> we are in a framework where colliding particles only collide with non-representative particles
            representative_particles[colliding_particle1].mass += representative_particles[colliding_particle2].mass

            # the new number of particles is set by dividing the representative particle group's total mass (which stays constant over the course of the simulation) by the new mass of each particle in that group 
            representative_particles[colliding_particle1].num_par = representative_particle_mass / representative_particles[colliding_particle1].mass

            collision_rates_matrix[:,colliding_particle1] = representative_particles[colliding_particle1].num_par * 0.5 * (representative_particles[colliding_particle1].mass + np.array([colliding_particle1.mass for colliding_particle1 in representative_particles])) / vol
            collision_rates_matrix[colliding_particle1, :] = np.array([p.num_par for p in representative_particles]) * 0.5 * (representative_particles[colliding_particle1].mass + np.array([p.mass for p in representative_particles])) / vol

        
        # we can calculate the new 1d vector corresponding the total collision rate propensities of each particle
            particle_collision_rates = np.sum(collision_rates_matrix, axis=1)


        # only update particle collision rates after all particle reactions have been applied
        # for all the particles that had collisions, update their collision rate matrix


        # for p in representative_colliding_particle1_list:
        # # we also need to update the collision rates matrix --> all collisions involving the first colliding particle need to be updated to account for the new mass and number of particles in that representative particle group
        #     collision_rates_matrix[:, p] = representative_particles[p].num_par * 0.5 * (representative_particles[p].mass + np.array([p.mass for p in representative_particles])) / vol
        # # we can calculate the new 1d vector corresponding the total collision rate propensities of each particle
        # particle_collision_rates = np.sum(collision_rates_matrix, axis=1)
        
        # update time
        sim_time += dtime

        # we print out the output histograms at certain intervals
        if sim_time > tout:
            # first initialize everything as zero
            mass_densities = np.zeros(nbins)

            # for each representative particle, we see what mass bin the masses of those representative particles fall in. 
            # So each representative particle corresponds to a set number of particles all with the same mass. To get the densities we 
            for p in representative_particles:
                bin_idx = np.searchsorted(mass_histogram, p.mass) - 1
                if bin_idx < nbins:
                    # we square the mass in fitting with A. Zsom and C. P. Dullemond (2008) when calculating densities
                    mass_densities[bin_idx] += (p.num_par * p.mass ** 2) / ((mass_histogram[bin_idx + 1] - mass_histogram[bin_idx]) * representative_particle_mass * ntot)
            
            # save file and print out to terminal
            filename = f"/Users/calebpainter/Downloads/Classes/AM207/final_project/outputs/R_leaping_nparticles_{ntot}_{num_reactions_leap}_reactions_python_output-{int(tout)}_run_{run}.txt"
            print(filename)
            with open(filename, "w") as f:
                for j in range(nbins):
                    midpoint = math.sqrt(mass_histogram[j] * mass_histogram[j + 1])
                    f.write(f"{midpoint} {mass_densities[j]}\n")
            tout += dtout

    end_time = time.time()

    # Calculate the runtime for this iteration
    iteration_runtime = end_time - start_time
    print(f"Simulation complete in {iteration_runtime:.6f} seconds.")
    print(f"{tot_num_collisions} collisions occured.")
    return iteration_runtime

if __name__ == "__main__":

    tot_particles = np.arange(10,320,20)
    runs = np.arange(0,30)
    particle_run_times = []
    particle_run_times_errors = []
    for tp in tot_particles:
        print(f'Running with {tp} particles')
        particle_times = []
        for run in runs:
            time_elapsed = main(int(tp),run)
            particle_times.append(time_elapsed)
        particle_run_times.append(np.mean(particle_times))
        particle_run_times_errors.append(np.std(particle_times)/np.sqrt(len(particle_times)))

    # np.save('/Users/calebpainter/Downloads/Classes/AM207/final_project/outputs/R_runtimes2', particle_run_times)
    # np.save('/Users/calebpainter/Downloads/Classes/AM207/final_project/outputs/R_runtimes_errors2', particle_run_times_errors)