#include<utils/random.hpp>

Random::Random(std::vector<std::uint64_t> & t_seeds, const std::uint64_t t_num_discard) : Random() {
	// seeds the random generator and advances its state
	if (t_seeds.size() == 0)
		generateRandomSeeds(t_seeds);
	seedRandomGenerator(t_seeds, t_num_discard);
}

void Random::generateRandomSeeds(std::vector<std::uint64_t> & t_seeds) {
	// draws a set of original seeds for random generator */
	t_seeds.resize(std::mt19937::state_size);
	std::random_device random;
	for (std::uint64_t i = 0; i < std::mt19937::state_size / 2; i++) {
		t_seeds[2 * i] = static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		t_seeds[2 * i + 1] = random();
	}
	random_count = 0;
	return;
}

void Random::seedRandomGenerator(const std::vector<std::uint64_t> & t_seeds, const std::uint64_t t_num_discard) {
	std::seed_seq seed_sequence(t_seeds.begin(), t_seeds.end());
	random_generator.seed(seed_sequence);
	has_spare_normal = false;
	random_count = 0;
	if (t_num_discard != 0) {
		discardRandom(t_num_discard);
		random_count = t_num_discard;
	}
	return;
}

void Random::drawRandomSeeds(std::vector<std::uint64_t> & t_seeds) {
	// draws new seeds from previously initialized generator and reseeds
	std::generate(t_seeds.begin(), t_seeds.end(), std::ref(random_generator));
	std::seed_seq seed_sequence(t_seeds.begin(), t_seeds.end());
	return;
}

const std::uint64_t Random::getRandom(void) {
	// increment random count and return a single random double in [0, 1).
	random_count++;
	return random_generator();
}

const double Random::getRandomUniform(void) {
	// generate unbiased, uniformly distributed doubles using 52 bits
    std::uint64_t r_u = (getRandom() & 0xFFFFFFFFFFFFFull) | 0x3FF0000000000000ull;
	double * r_d = reinterpret_cast<double *>(&r_u);
	return * r_d - 1.;
}

void Random::discardRandom(const std::uint64_t t_num_discard) {
	// add to the random count and discard values from the global RNG.
	random_generator.discard(t_num_discard);
	random_count += t_num_discard;
}

void Random::getRandomUnitNormalPair(std::pair<double, double> & t_pair) {
	// samples from a normal distribution using Box-Muller transformation
	double r = sqrt(-2. * log(getRandomUniform()));
	double theta = two_pi * getRandomUniform();
	if (has_spare_normal) {
		t_pair.first = spare_normal;
		t_pair.second = r * cos(theta); 
		spare_normal = r * sin(theta);
	}
	else {
		t_pair.first = r * cos(theta); 
		t_pair.second = r * sin(theta);
	}
	return;
}

const double Random::getRandomUnitNormal(void) {
	// samples from a normal distribution using Box-Muller transformation
	if (has_spare_normal) {
		has_spare_normal = false;
		return spare_normal;
	}
	has_spare_normal = true;
	double r = sqrt(-2. * log(getRandomUniform()));
	double theta = two_pi * getRandomUniform();
	spare_normal = r * sin(theta);
	return r * cos(theta);	
}

const double Random::getRandomNormal(const double t_mean, const double t_standard_deviation) {
	// samples from a normal distribution using Box-Muller transformation
	return getRandomUnitNormal() * t_standard_deviation + t_mean;	
}

const std::uint64_t Random::getRandomPoisson(const double t_mean) {
	// samples Poisson distribution using inversion sampling
	double u = getRandomUniform();
	double p = exp(-t_mean);
	double s = p;
	std::uint64_t x = 0;
	while (u > s) {
		x++;
		p *= t_mean / ((double) x);
		s += p;
	}
	return x;
}

const std::uint64_t Random::getRandomCount(void) {
	// returns values of private count for the number of random numbers used so far
	return random_count;
}
