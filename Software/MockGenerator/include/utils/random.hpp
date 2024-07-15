//************************
//*
//*
//*
//*
//************************

#ifndef RANDOM_H
#define RANDOM_H

#include<li_class.hpp>

#include<chrono>
#include<random>
#include<algorithm>

class Random : public LIClass {

	public :

	Random(void){exception_prefix = "Error::Random, ";};
	Random(std::vector<std::uint64_t> & t_seeds, const std::uint64_t t_num_discard = 0);

	void seedRandomGenerator(const std::vector<std::uint64_t> & t_seeds, const std::uint64_t t_num_discard = 0);
	void drawRandomSeeds(std::vector<std::uint64_t> & t_seeds);
	const std::uint64_t getRandom(void);
	const double getRandomUniform(void);
	void discardRandom(const std::uint64_t t_num_discard);
	void getRandomUnitNormalPair(std::pair<double, double> & t_pair);
	const double getRandomUnitNormal(void);
	const double getRandomNormal(const double t_mean, const double t_standard_deviation);
	const std::uint64_t getRandomPoisson(const double t_mean);
	const std::uint64_t getRandomCount(void);

	std::mt19937_64 random_generator;
	std::uint64_t random_count;
	bool has_spare_normal = false;
	double spare_normal;
	void generateRandomSeeds(std::vector<std::uint64_t> & t_seeds);
    const double two_pi = 2. * acos(-1.);

};

#endif // *** RANDOM_H *** //
