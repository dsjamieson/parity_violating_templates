//************************
//*
//*
//*
//*
//************************

#ifndef TASK_DISTRIBUTOR_H
#define TASK_DISTRIBUTOR_H

#include<li_class.hpp>

#include<vector>
#include<algorithm>

template<typename t_type, typename t_it_type = t_type> 
class TaskDistributor : public LIClass {
	
	public :

	TaskDistributor(const std::uint64_t t_total_num_tasks, const std::uint64_t t_start = 0, const std::uint64_t t_num_threads = 0) {
		exception_prefix = "Error::TaskDistributor, ";
		const std::uint64_t num_threads = (t_num_threads == 0) ? static_cast<std::uint64_t>(omp_get_max_threads()) : t_num_threads;
		const std::uint64_t final_task = t_start + t_total_num_tasks;
		const std::uint64_t id = static_cast<std::uint64_t>(omp_get_thread_num());
		const std::uint64_t remainder = t_total_num_tasks % num_threads;
		num_tasks = (id < remainder) ? t_total_num_tasks / num_threads + 1 : t_total_num_tasks / num_threads;
		start = (id < remainder) ? t_start + id * num_tasks : t_start + id * num_tasks + remainder;
		stop = std::min(start + num_tasks, final_task);
		worker = (start < final_task) ? true : false;
		first_worker = (id == 0) ? true : false;
		last_worker = (worker && stop == final_task) ? true : false;
	};

	TaskDistributor(std::vector<t_type> & t_tasks, const std::uint64_t t_stride = 1, const std::uint64_t t_num_threads = 0) : TaskDistributor(t_tasks.size() / t_stride, t_num_threads) {
        stride = t_stride;
        begin = t_tasks.begin() + start;
        next = begin + t_stride;
        end = t_tasks.begin() + stop;
        if (last_worker)
            end = t_tasks.end();
    }

	bool worker, first_worker, last_worker;
	std::uint64_t num_tasks, start, stop, stride;
    typename std::vector<t_it_type>::iterator begin;
    typename std::vector<t_it_type>::iterator end;
    typename std::vector<t_it_type>::iterator next;

    void advance(void) {
        std::advance(begin, stride);
        std::advance(next, stride);
        next = (next > end) ? end : next;
        return;
    }

};

template<typename t_type, typename t_it_type = t_type> 
class ConstTaskDistributor : public LIClass {
	
	public :

	ConstTaskDistributor(const std::uint64_t t_total_num_tasks, const std::uint64_t t_start = 0, const std::uint64_t t_num_threads = 0) {
		exception_prefix = "Error::ConstTaskDistributor, ";
		std::uint64_t num_threads = (t_num_threads == 0) ? static_cast<std::uint64_t>(omp_get_max_threads()) : t_num_threads;
		const std::uint64_t final_task = t_start + t_total_num_tasks;
		const std::uint64_t id = static_cast<std::uint64_t>(omp_get_thread_num());
		const std::uint64_t remainder = t_total_num_tasks % num_threads;
		num_tasks = (id < remainder) ? t_total_num_tasks / num_threads + 1 : t_total_num_tasks / num_threads;
		start = (id < remainder) ? t_start + id * num_tasks : t_start + id * num_tasks + remainder;
		stop = std::min(start + num_tasks, final_task);
		worker = (start < final_task) ? true : false;
		first_worker = (id == 0) ? true : false;
		last_worker = (worker && stop == final_task) ? true : false;
	};

	ConstTaskDistributor(const std::vector<t_type> & t_tasks, const std::uint64_t t_stride = 1, const std::uint64_t t_num_threads = 0) : ConstTaskDistributor(t_tasks.size() / t_stride, t_num_threads) {
        stride = t_stride;
        begin = t_tasks.begin() + start;
        next = begin + t_stride;
        end = t_tasks.begin() + stop;
        if (last_worker)
            end = t_tasks.end();
    }

	bool worker, first_worker, last_worker;
	std::uint64_t num_tasks, start, stop, stride;
    typename std::vector<t_it_type>::const_iterator begin;
    typename std::vector<t_it_type>::const_iterator end;
    typename std::vector<t_it_type>::const_iterator next;

    void advance(void) {
        std::advance(begin, stride);
        std::advance(next, stride);
        next = (next > end) ? end : next;
        return;
    }

};

#endif // *** TASK_DISTRIBUTOR_H *** //
