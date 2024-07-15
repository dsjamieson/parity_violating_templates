//************************
//*
//*
//*
//*
//************************

#ifndef FIELD_H
#define FIELD_H

#include<li_class.hpp>
#include<utils/task_distributor.hpp>
#include<vector>
#include<ranges>
#include<complex>
#include<numeric>
#include<cmath>
#include<fftw3-mpi.h>

class Field : public LIClass {

    public:

    Field(void){
		exception_prefix = "Error::Field, ";
		label_string = "field";
	}

    Field(const std::uint64_t t_num_mesh_1d, const double t_box_length) : Field() {
        initialize(t_num_mesh_1d, t_box_length);
    }

    Field(const Field & t_field) : Field(t_field.num_mesh_1d, t_field.box_length) {}

    void initialize(const std::uint64_t t_num_mesh_1d, const double t_box_length);

    std::uint64_t num_mesh_1d;
    std::uint64_t half_num_mesh_1d;
    std::uint64_t num_modes_last_d;
    std::uint64_t two_num_modes_last_d;
	std::uint64_t num_mesh_sites;
    std::uint64_t local_size;
    std::uint64_t local_num_x;
    std::uint64_t local_start_x;
    std::uint64_t local_end_x;
	std::uint64_t num_padding_1d;

    double box_length;
    double box_volume;
    double bin_length;
    double bin_volume;
    double fundamental_wave_number;
    double nyquist_wave_number;
    double forward_dft_normalization;
    double backward_dft_normalization;
    std::vector<double> wave_numbers;

    const bool sizesAreConsistent(const Field & t_mesh_1, const Field & t_mesh_2) const;
    const bool sizesAreConsistent(const Field & t_mesh) const;

    const std::uint64_t getMeshIndex(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const;
    const std::uint64_t getLocalMeshIndex(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const;
    const std::uint64_t getModeIndex(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const;
    const std::uint64_t getLocalModeIndex(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const;
	const std::uint64_t getFieldIndex(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const;
	const std::uint64_t getLocalFieldIndex(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const;

};

#endif // *** FIELD_H *** //
