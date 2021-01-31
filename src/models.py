import time
import numpy as np
import random
import subprocess
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape


# --------------- Different types of mutation algorithms  ------------------------

# Random Resetting is an extension of the bit flip for the integer representation.
# In this, a random value from the set of permissible values is assigned to a randomly chosen gene.
def random_resetting_mutation(phenotype: np.ndarray, password_cracker):
    phenotype_index = random.randint(0, phenotype.shape[0] - 1) 
    phenotype[phenotype_index] = password_cracker.random_domain_val()
    
    #print(phenotype)
    #print("")
    return phenotype


# In swap mutation, we select two positions on the chromosome at random, and interchange the values.
def swap_mutation(phenotype: np.ndarray, password_cracker):
    swaps = np.random.randint(0, phenotype.shape[0], size=2)
    phenotype[swaps[0]], phenotype[swaps[1]] = phenotype[swaps[1]], phenotype[swaps[0]]
    
    #print(phenotype)
    #print("")
    return phenotype


# In this, from the entire chromosome, a subset of genes is chosen and their values are scrambled or shuffled randomly.
def scramble_mutation():
    pass


# In inversion mutation, we select a subset of genes like in scramble mutation,
# but instead of shuffling the subset, we merely invert the entire string in the subset.
def inversion_mutation():
    pass

def close_random_mutation(phenotype: np.ndarray,password_cracker):
    phenotype_index = random.randint(0, phenotype.shape[0] - 1)
    rand 
    phenotype[phenotype_index] = password_cracker.random_domain_val()



# adds or removes  one caracter or both
def expend_retract_mutation(phenotype: np.ndarray, password_cracker):
    rand = random.random()
    #print(rand)
    if rand <= 0.5 and phenotype.shape[0] > password_cracker.domain_min_size :
        #print("1")
        phenotype_index = random.randint(0, phenotype.shape[0] - 1) 
        #print("bjr")
        phenotype = np.concatenate([phenotype[:phenotype_index],phenotype[phenotype_index+1:]])
        #print(phenotype)
        #print("")
        return phenotype

    if rand <=0.5 and phenotype.shape[0] < password_cracker.domain_max_size :
        #print("2")
        rand_char = random.choice(password_cracker.phenotype_domain)
        #print("bjr")
        phenotype_index = random.randint(0, phenotype.shape[0] - 1)
        phenotype =  np.concatenate([phenotype[:phenotype_index], [rand_char],phenotype[phenotype_index:]])
       # print(arr)
       # print("bjr")
       # print("")
        return phenotype

    elif rand > 0.5:
        #print("3")
        phenotype_index = random.randint(0, phenotype.shape[0] - 1)
        #print("bjr")
        phenotype = np.concatenate([phenotype[:phenotype_index],phenotype[phenotype_index+1:]])
        #print(pheno)
        #print("bjrrr")
        rand_char = random.choice(password_cracker.phenotype_domain)
        phenotype_index = random.randint(0, phenotype.shape[0] - 1)
        phenotype = np.concatenate([phenotype[:phenotype_index], [rand_char],phenotype[phenotype_index:]])
        #print(phenotype)
        #print("")
        return phenotype

# --------------- Crossover algorithms ------------------------

# In this one-point crossover, a random crossover point is selected
# and the tails of its two parents are swapped to get new off-springs.
def one_point_crossover(parent1, parent2, child1, child2, password_cracker):
    p1 = random.randint(password_cracker.domain_min_size, parent1.shape[0])
    p2 = random.randint(password_cracker.domain_min_size, parent2.shape[0])

    child1 = np.concatenate([child1[:p2], parent2[p2:]])
    child2 = np.concatenate([child2[:p1], parent1[p1:]])

    return child1, child2


# Multi point crossover is a generalization of the one-point
# crossover wherein alternating segments are swapped to get new off-springs.
def multi_point_crossover(parent1, parent2, child1,child2,password_cracker):
    size = min(parent1.shape[0], parent2.shape[0])
  
    mid_p1 = random.randint(0, int(size/2))
    mid_p2 = random.randint(int((size/2))+1, size)
    #print(mid_p1)
    #print(mid_p2)
    #print(mid_p1)

    child1 = np.concatenate([parent1[:mid_p1],parent2[mid_p1:mid_p2],parent1[mid_p2:]])
    child2 = np.concatenate([parent2[:mid_p1],parent1[mid_p1:mid_p2],parent2[mid_p2:]])

    return child1,child2



# In a uniform crossover, we don’t divide the chromosome into segments,
# rather we treat each gene separately. In this, we essentially flip a coin
# for each chromosome to decide whether or not it’ll be included in the off-spring.
# We can also bias the coin to one parent, to have more genetic material in the child from that parent.
def uniform_crossover(parent1, parent2, child1, child2, password_cracker):
    min_len = min(parent1.shape[0], parent2.shape[0])

    for i in range(min_len):
        if np.random.rand() < 0.50:
            child1[i] = parent2[i]
            child2[i] = parent1[i]
        else:
            child2[i] = parent2[i]
            child1[i] = parent1[i]

    return child1, child2


# works by taking the weighted average of the two parents by using the following formulae −
#     Child1 = α.x + (1-α).y
#     Child2 = α.x + (1-α).y
def whole_arithmetic_recombination_crossover():
    pass



class PasswordCracker:

    def __init__(self, student_id: str, possible_values: list, pass_min_size: int, pass_max_size: int, parametres : list):
        self.student_id = student_id
        self.phenotype_domain = np.array(possible_values)
        self.domain_min_size = pass_min_size
        self.domain_max_size = pass_max_size
        self.scores = []
        self.step = 0
        self.chance_resetting = parametres[0]
        self.chance_expend = parametres[1]
        self.chance_swap_mutation = parametres[2]
        self.good_genes_keep = parametres[3]
        self.bad_genes_keep = parametres[4]
        self.crossover_active = True

    def check(self, phenotypes: np.ndarray) -> np.ndarray:
        #print(phenotypes)
        passwords = [''.join(phenotype) for phenotype in phenotypes]
        #print(passwords)
        results = np.zeros(len(passwords), dtype=np.float32)
        args = np.concatenate((["./unlock_mac", str(self.student_id)], passwords))
        with subprocess.Popen(args, stdout=subprocess.PIPE) as proc:
            i = 0
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                results[i] = float(str(line).split("\\t")[1].split("\\n")[0])
                i += 1

        return results

    def random_domain_val(self):
        return random.choice(self.phenotype_domain)

    # return a list of size nb_phenotypes of random phenotypes
    def generate_random_phenotypes(self, nb_phenotypes) -> np.ndarray:
        phenotypes_sizes = np.random.randint(low=self.domain_min_size, high=self.domain_max_size + 1, size=nb_phenotypes)
        phenotypes = np.empty(nb_phenotypes, dtype=np.object)
        #print(nb_phenotypes)
        #print("Done")
        for i, size in enumerate(phenotypes_sizes):
            # replace permet d'avoir des repétitions des chars dans les phenotypes
            phenotypes[i] = np.random.choice(self.phenotype_domain, size=size, replace=True)

        return phenotypes

    def record_scores(self, scores):
        self.scores[self.step][0] = 100 * np.max(scores)
        self.scores[self.step][1] = 100 * np.mean(scores)
        self.scores[self.step][2] = 100 * np.min(scores)

    def select_parents(self, population: np.ndarray) -> np.ndarray:

        population_scores = self.check(population)
        probability_scores = population_scores / np.max(population_scores)
        population_indecies = np.arange(0, population.shape[0])
        
        parents_indecies = random.choices(population_indecies, weights=probability_scores, k=2)
        parents = population[parents_indecies]

        return parents

    def elitism_selection(self, population: np.ndarray,good_genes_size,bad_genes_size):
        
        #print(population)
        population_scores = self.check(population)
        #print(population_scores.shape[0])
        population_indecies = np.arange(0, population.shape[0])
        sorted_population_indecies = np.argsort(-(population_scores))
        

        random_indecies = random.choices(population_indecies,k=bad_genes_size)
       # print(random_indecies)
        random_pop = population[random_indecies]
        random_scores = population_scores[random_indecies]
        #print(random_scores)
        #print("Random")
        population = population[sorted_population_indecies]
        good_pop = population[:good_genes_size]
        good_pop_indecies = np.arange(0, good_pop.shape[0])
        good_pop_scores = population_scores[good_pop_indecies]
        #print(good_pop_scores)
        #print("Good")


        result = np.concatenate((good_pop,random_pop))
        result_score = np.concatenate((good_pop_scores,random_scores))
        #print(result_score)
        #print("All")

        self.record_scores(result_score)
       
        return result
    #TODO FAIRE COPIE AVANT DE CONTINUER 
    #TODO  corriger fonctionnement rand ne marche pas sur grand nombreÒ
    def rank_selection(self,parents):
        #print("Parent")
        #print(parents)
        parents_scores = self.check(parents)
        #print("Score")
        #print(parents_scores)
        size = np.size(parents)
        factor = 2/((size)*(size-1))
       
        sorted_population_indecies = np.argsort(parents_scores)
        parents_scores_sorted = parents_scores[sorted_population_indecies]
        parents_sorted = parents[sorted_population_indecies]
        #parents_sorted_inverses = parents[np.argsort(-(parents_scores))]


        for i , parent in enumerate(parents_scores_sorted):
            if i == 0:
                parents_scores_sorted[i] = factor*i
            else : parents_scores_sorted[i] = factor*i + parents_scores_sorted[i-1]    
        
        return random.choices(parents_sorted,parents_scores_sorted,k=2)    
        """ rand = random.random()
        #print("Rand")
        #print(rand)

        for i , parent in enumerate(parents_scores_sorted):
            if(rand <= parent):
                #print((size-1)-i)
                #print(parents_sorted_inverses[(size-1)-i])
                return parents_sorted_inverses[(size-1)-i] """
    

        
                 

    def mutations(self, phenotype: np.ndarray):

        if random.random() < self.chance_swap_mutation:
            return swap_mutation(phenotype, self)
        elif random.random() < self.chance_resetting:
            return random_resetting_mutation(phenotype, self)
        elif random.random() < self.chance_expend:
            return expend_retract_mutation(phenotype, self)
        else: 
            return phenotype

    def crossover(self, parent: np.ndarray, other_parent: np.ndarray, x: int):
        child1 = np.copy(parent)
        child2 = np.copy(other_parent)
        
        # if random.random() <= 0.99:
        if(x == 1):
            child1,child2 = uniform_crossover(parent, other_parent, child1,child2,self)
        elif (x == 2):    
            child1,child2 = multi_point_crossover(parent, other_parent,child1,child2,self)
        # else:
        # child1, child2 = one_point_crossover(parent, other_parent, child1, child2, self)

        return child1, child2

    def next_gen(self, parents,size):
        #print(self.phenotype_domain[5])
        #print(parents.shape[0])
        #print(self.rank_selection(parents))
        new_population = np.empty((size - parents.shape[0]), dtype=np.object)
        #print(new_population.shape[0])
        chi = 0
        x= 1
        desired_len = size - np.size(parents)
        while chi <desired_len:
            #parent,other_parent = self.rank_selection(parents)
            parent,other_parent = self.select_parents(parents)
            if(self.crossover_active):
                child1, child2 = self.crossover(parent, other_parent,x)
                #print(child1)
                #print(child2)
            else:    
                child1,child2 = self.crossover(parent,other_parent,3)
                #print(child1)
                #print(child2)    

            if(x==1):
                x=2
            else :
                x=1    

            try:
                child1 = self.mutations(child1)
                child2 = self.mutations(child2)
            except Exception as e:
                print("Erreur")
                print(parent)
                print(other_parent)
                print(child1)
                print(child2)
                break

            new_population[chi] = child1
            chi += 1
            new_population[chi] = child2
            chi += 1
            #print("CHI " + str(chi))
        #print(new_population)
        parents = np.concatenate([parents,new_population])
        return parents

    def print_infos(self, parents, population, step):
        scores = self.check(parents)
        print('max', np.max(scores))
        print('min', np.min(scores))
        print('mean', np.mean(scores))
        print('parents', parents.shape)
        print('next gen', population.shape)
        # np.savetxt(f'data/pass{step}.txt', population[indices])
        res = [ ' '.join(p) for p in parents[:5] ]
        for r in res:
            print(r)

        print()
        print()
        # input('next?')

    def run(self, init_population_size: int,steps: int, interval: int = 50):
        population = self.generate_random_phenotypes(init_population_size)
        self.step = 0
        self.scores = np.zeros(shape=(steps, 3), dtype=np.int8)

        start = time.time()

        while True:
            #self.select_parents(population,10)
            #print(population.shape[0])
            parents = self.elitism_selection(population, int(self.good_genes_keep*init_population_size),int(self.bad_genes_keep*init_population_size))

            if self.step % interval == 0:
                print(f'step: {self.step}')
                print('time', (time.time() - start) // 60, 'min :', round((time.time() - start) % 60, 2), 's')
                self.print_infos(parents, population, self.step)

            self.step += 1
            if self.step >= steps:
                break

           

            population = self.next_gen(parents,init_population_size)

        self.show_score()


    def show_score(self):
        fig, axs = plt.subplots(3)
        labels = ['Max score', 'Mean score', 'Min score']
        for i in range(3):
            axs[i].plot(self.scores[:, i])
            axs[i].set_title(labels[i])

        plt.show()
