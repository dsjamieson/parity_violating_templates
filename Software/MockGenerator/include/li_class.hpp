//************************
//*
//*
//*
//*
//************************

#ifndef LI_CLASS_H
#define LI_CLASS_H

#include<stdexcept>
#include<iostream>
#include<mpi.h>
#include<omp.h>
#include<chrono>
#include<vector>

class LIException : public std::runtime_error {

	public :

	LIException(void) : std::runtime_error(""){};

	std::string what_message;

	const char * what(void) const noexcept override {
		return what_message.c_str();
	}

};

class LITimer {

	public :

	LITimer(void) {
		clear();
	}

	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::duration duration;
	std::chrono::steady_clock::duration cumulative_duration;

	void clear(void) {
		cumulative_duration = std::chrono::steady_clock::duration::zero();
		return;
	}

	void setStart(void) {
		start_time = std::chrono::steady_clock::now();
		return;
	}

	void setDuration(void) {
		duration = std::chrono::steady_clock::now() - start_time;
		cumulative_duration += duration;
		return;
	}

	template<typename t_type = std::chrono::microseconds>
	std::uint64_t getDuration(void) const{
		return std::chrono::duration_cast<t_type>(duration).count();
	}

	template<typename t_type = std::chrono::microseconds>
	std::uint64_t getCumulativeDuration(void) const {
		return std::chrono::duration_cast<t_type>(cumulative_duration).count();
	}

};

class LIClass {

	public :

	LIClass(void){};

	std::string exception_prefix = "Error::LIClass, ";
	std::string label_string = "li class";
	LIException exception;

	std::vector<LITimer> _timers;

	void assert(const bool t_assertion, const std::string t_message = "") {
		if (!t_assertion) {
			exception.what_message += exception_prefix + t_message + "\n";
			throw exception;
		}
		return;
	}

	void exit(const std::int64_t t_exit_code = 0) const {
		MPI_Finalize();
		std::exit(t_exit_code);
	}

	const std::uint64_t getNumNodes(void) const {
		int num_nodes;
		MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
		return static_cast<std::uint64_t>(num_nodes);
	};

	const std::uint64_t getNodeID(void) const {
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		return static_cast<std::uint64_t>(rank);
	};

	const std::uint64_t getNumThreads(void) const {
		int num_threads = omp_get_max_threads();
		return static_cast<std::uint64_t>(num_threads);
	};

};

#endif // *** LI_CLASS_H *** //
