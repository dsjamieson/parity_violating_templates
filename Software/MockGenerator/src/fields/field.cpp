#include<fields/field.hpp>

void Field::initialize(const std::uint64_t t_num_mesh_1d, const double t_box_length) {
	num_mesh_1d = t_num_mesh_1d;
	num_mesh_sites = num_mesh_1d * num_mesh_1d * num_mesh_1d;
	half_num_mesh_1d = num_mesh_1d / 2;
	num_modes_last_d = half_num_mesh_1d + 1;
	two_num_modes_last_d = 2 * num_modes_last_d;
	num_padding_1d = (num_mesh_1d % 2 == 0) ? 2 : 1;
	box_length = t_box_length;
	box_volume = box_length * box_length * box_length;
	bin_length = box_length / static_cast<double>(num_mesh_1d);
	bin_volume = bin_length * bin_length * bin_length;
	fundamental_wave_number = 2. * acos(-1.) / box_length;
	nyquist_wave_number = fundamental_wave_number * half_num_mesh_1d;
	try {
		wave_numbers.resize(num_mesh_1d);
	}
	catch(...) {
		assert(false, "failed to allocate scalar field mesh wave numbers");
	}
	for (std::uint64_t i = 0; i < wave_numbers.size(); i++) {
		wave_numbers[i] = (i > half_num_mesh_1d) ? static_cast<double>(static_cast<std::int64_t>(i) - static_cast<std::int64_t>(num_mesh_1d)) :
			static_cast<double>(i);
		wave_numbers[i] *= fundamental_wave_number;
	}
	{
		long local_num_x_tmp;
		long local_start_x_tmp;
		local_size = fftw_mpi_local_size_3d(num_mesh_1d, num_mesh_1d, two_num_modes_last_d, MPI_COMM_WORLD, &local_num_x_tmp, &local_start_x_tmp);
		local_num_x = static_cast<std::uint64_t>(local_num_x_tmp);
		local_start_x = static_cast<std::uint64_t>(local_start_x_tmp);
	}
	local_end_x = local_start_x + local_num_x;
	return;
}

const bool Field::sizesAreConsistent(const Field & t_field_1, const Field & t_field_2) const {
	return t_field_1.num_mesh_1d == t_field_2.num_mesh_1d;
}

const bool Field::sizesAreConsistent(const Field & t_field) const {
	return sizesAreConsistent(*this, t_field);
}

const std::uint64_t Field::getMeshIndex(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const {
	return t_k + two_num_modes_last_d * (t_j + num_mesh_1d * t_i);
}

const std::uint64_t Field::getLocalMeshIndex(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const {
	return t_k + two_num_modes_last_d * (t_j + num_mesh_1d * (t_i - local_start_x));
}

const std::uint64_t Field::getModeIndex(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const {
	return t_k + num_modes_last_d * (t_j + num_mesh_1d * t_i);
}

const std::uint64_t Field::getLocalModeIndex(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const {
	return t_k + num_modes_last_d * (t_j + num_mesh_1d * (t_i - local_start_x));
}

const std::uint64_t Field::getFieldIndex(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const {
	return t_k + num_mesh_1d * (t_j + num_mesh_1d * t_i);
}

const std::uint64_t Field::getLocalFieldIndex(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const {
	return t_k + num_mesh_1d * (t_j + num_mesh_1d * (t_i - local_start_x));
}
