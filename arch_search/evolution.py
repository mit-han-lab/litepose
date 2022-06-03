import copy
import random
import numpy as np
import time
from arch_manager import ArchManager

__all__ = ['EvolutionFinder']

def rand(c):
	return random.randint(0, c - 1)

class EvolutionFinder:

	def __init__(self, cfg, efficiency_predictor, accuracy_predictor, **kwargs):
		self.cfg = cfg

		self.efficiency_predictor = efficiency_predictor
		self.accuracy_predictor = accuracy_predictor
		self.arch_manager = ArchManager(cfg)

		self.mutate_prob = kwargs.get('mutate_prob', 0.1)
		self.population_size = kwargs.get('population_size', 40)
		self.max_time_budget = kwargs.get('max_time_budget', 40)
		self.parent_ratio = kwargs.get('parent_ratio', 0.25)
		self.mutation_ratio = kwargs.get('mutation_ratio', 0.5)

	def set_efficiency_constraint(self, new_constraint):
		self.efficiency_constraint = new_constraint

	def random_sample(self):
		constraint = self.efficiency_constraint
		while True:
			sample = self.arch_manager.random_sample()
			efficiency = self.efficiency_predictor.predict_eff(sample)
			if efficiency <= constraint:
				return sample, efficiency

	def mutate_sample(self, sample):
		constraint = self.efficiency_constraint
		while True:
			cfg_arch = copy.deepcopy(sample)
			if random.random() < self.mutate_prob:
				img_size = 256 + 64 * rand(5)
				cfg_arch['img_size'] = img_size
			if random.random() < self.mutate_prob:
				cfg_arch['input_channel'] = self.arch_manager.rand_channel(self.arch_manager.input_channel)
			for i in range(len(self.arch_manager.deconv_setting)):
				if random.random() < self.mutate_prob:
					cfg_arch['deconv_setting'][i] = self.arch_manager.rand_channel(self.arch_manager.deconv_setting[i])
			for i in range(len(self.arch_manager.arch_setting)):
				if random.random() < self.mutate_prob:
					c, n, s = self.arch_manager.arch_setting[i]
					cfg_arch['backbone_setting'][i]['channel'] = self.arch_manager.rand_channel(c)
			efficiency = self.efficiency_predictor.predict_eff(cfg_arch)
			if efficiency <= constraint:
				return cfg_arch, efficiency

	def crossover_sample(self, sample1, sample2):
		constraint = self.efficiency_constraint
		while True:
			new_sample = copy.deepcopy(sample1)
			for key in new_sample.keys():
				if not isinstance(new_sample[key], list):
					continue
				for i in range(len(new_sample[key])):
					new_sample[key][i] = random.choice([sample1[key][i], sample2[key][i]])

			efficiency = self.efficiency_predictor.predict_eff(new_sample)
			if efficiency <= constraint:
				return new_sample, efficiency

	def run_evolution_search(self, verbose=False):
		"""Run a single roll-out of regularized evolution to a fixed time budget."""
		max_time_budget = self.max_time_budget
		population_size = self.population_size
		mutation_numbers = int(round(self.mutation_ratio * population_size))
		parents_size = int(round(self.parent_ratio * population_size))
		constraint = self.efficiency_constraint

		best_valids = [-100]
		population = []  # (validation, sample, latency) tuples
		child_pool = []
		efficiency_pool = []
		accuracy_pool = []
		best_info = None
		if verbose:
			print('Generate random population...')
		for _ in range(population_size):
			print(_)
			sample, efficiency = self.random_sample()
			child_pool.append(sample)
			efficiency_pool.append(efficiency)
			accuracy_pool.append(self.accuracy_predictor.predict_acc(sample))

		for i in range(population_size):
			population.append((accuracy_pool[i], child_pool[i], efficiency_pool[i]))

		if verbose:
			print('Start Evolution...')
		# After the population is seeded, proceed with evolving the population.
		for iter in range(max_time_budget):
			parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
			acc = parents[0][0]
			if verbose:
				print('Iter: {} Acc: {}'.format(iter, parents[0][0]))

			if acc > best_valids[0]:
				best_valids = parents[0]

			population = parents
			child_pool = []
			efficiency_pool = []
			accuracy_pool = []

			for i in range(mutation_numbers):
				par_sample = population[np.random.randint(parents_size)][1]
				# Mutate
				new_sample, efficiency = self.mutate_sample(par_sample)
				child_pool.append(new_sample)
				efficiency_pool.append(efficiency)
				accuracy_pool.append(self.accuracy_predictor.predict_acc(new_sample))

			for i in range(population_size - mutation_numbers):
				par_sample1 = population[np.random.randint(parents_size)][1]
				par_sample2 = population[np.random.randint(parents_size)][1]
				# Crossover
				new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2)
				child_pool.append(new_sample)
				efficiency_pool.append(efficiency)
				accuracy_pool.append(self.accuracy_predictor.predict_acc(new_sample))

			for i in range(population_size):
				population.append((accuracy_pool[i], child_pool[i], efficiency_pool[i]))

		return best_valids