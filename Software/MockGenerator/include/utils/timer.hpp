//********************
//*
//*
//*
//********************

#ifndef TIMER_H
#define TIMER_H

#include<li_class.hpp>
#include <chrono>
#include <iomanip>
#include <cinttypes>
#include <cmath>

class Timer : public LIClass {

	public:

    Timer(const std::uint64_t t_max_message_length = 0) {
		exception_prefix = "Error::Timer, ";
        start_time = std::chrono::steady_clock::now();
        last_time = std::chrono::steady_clock::now();
        max_message_length = t_max_message_length;
    }

    Timer(const std::string t_message) : Timer(t_message.size()) {}

	void setMaxMessageLength(const std::uint64_t t_max_message_length) {max_message_length = t_max_message_length;};

	void setMaxMessageLength(const std::string t_message) {max_message_length = t_message.size();};

    void printStart(const std::string t_message) {
        current_message_length = t_message.size();
        std::cout << t_message << std::flush;
        return;
    }

    void printDone(bool t_new_line = true, const std::uint64_t t_num_spaces = 0) {
        const std::chrono::steady_clock::time_point this_time = std::chrono::steady_clock::now();
        const double step_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(this_time - last_time).count()) * 1.e-3;
        const double cumulative_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(this_time - start_time).count()) * 1.e-3;
        const std::uint64_t step_time_h = static_cast<std::uint64_t>(std::floor(step_time / 60. / 60.));
        const std::uint64_t step_time_m = static_cast<std::uint64_t>(std::floor(step_time / 60. - step_time_h * 60));
        const double step_time_s = step_time - step_time_m * 60. - step_time_h * 60. * 60.;
        const std::uint64_t cumulative_time_h = static_cast<std::uint64_t>(std::floor(cumulative_time / 60. / 60.));
        const std::uint64_t cumulative_time_m = static_cast<std::uint64_t>(std::floor(cumulative_time / 60. - cumulative_time_h * 60));
        const double cumulative_time_s = cumulative_time - cumulative_time_m * 60. - cumulative_time_h * 60. * 60.;
        if (max_message_length == 0 || t_num_spaces != 0) {
            for (std::uint64_t i = 0; i < t_num_spaces; i++)
                std::cout << " ";
        }
        else if (max_message_length > current_message_length) {
            for (std::uint64_t i = 0; i < max_message_length - current_message_length; i++)
                std::cout << " ";
        }
        else
            std::cout << " ";
        std::cout << "done ";
        std::cout << "[" << std::fixed << std::setprecision(2) << std::setfill('0')
            << std::setw(2) << step_time_h << ":" << std::setw(2) << step_time_m << ":" << std::setw(5) << step_time_s << ", "
            << std::setw(2) << cumulative_time_h << ":" << std::setw(2) << cumulative_time_m << ":" << std::setw(5) << cumulative_time_s << "]";
		if (t_new_line)
			std::cout << std::endl;
        last_time = this_time;
        return;
    }

	std::chrono::steady_clock::time_point start_time;
	std::chrono::steady_clock::time_point last_time;
	std::uint64_t max_message_length;
	std::uint64_t current_message_length;

};

#endif // *** TIMER_H *** //
