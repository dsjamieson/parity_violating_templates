#include<fields/vector_field.hpp>

void ScalarField::initialize(const std::uint64_t t_num_mesh_1d, const double t_box_length) {
	try {
		Field::initialize(t_num_mesh_1d, t_box_length);
		_data.resize(local_size);
	}
	catch(...) {
		assert(false, "field to allocate " + label_string + " mesh");
	}
	mesh = reinterpret_cast<double * >(data());
	modes = reinterpret_cast<std::complex<double> * >(data());
	fftw_modes = reinterpret_cast<fftw_complex *>(data());
	forward_plan = fftw_mpi_plan_dft_r2c_3d(num_mesh_1d, num_mesh_1d, num_mesh_1d, mesh, fftw_modes, MPI_COMM_WORLD, FFTW_ESTIMATE); 
	backward_plan = fftw_mpi_plan_dft_c2r_3d(num_mesh_1d, num_mesh_1d, num_mesh_1d, fftw_modes, mesh, MPI_COMM_WORLD, FFTW_ESTIMATE);
	forward_dft_normalization = bin_volume;
	backward_dft_normalization = 1. / box_volume;
	return;
}

double & ScalarField::operator[](const std::uint64_t t_i) {
	return _data[t_i];
}

double * ScalarField::data(void) {
	return _data.data();
}

const std::vector<double>::iterator ScalarField::begin(void) {
	return _data.begin();
}

const std::vector<double>::const_iterator ScalarField::begin(void) const {
	return _data.begin();
}

const std::vector<double>::iterator ScalarField::end(void) {
	return _data.end();
}

const std::uint64_t ScalarField::size(void) const {
	return _data.size();
}

const double ScalarField::getField(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const {
	return mesh[(t_k + two_num_modes_last_d * (t_j + num_mesh_1d * (t_i - local_start_x)))];
}

const std::complex<double> ScalarField::getMode(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const {
	return modes[t_k + num_modes_last_d * (t_j + num_mesh_1d * (t_i - local_start_x))];
}

void ScalarField::transformDFT(ScalarField & t_modes, const bool t_norm) const {
	if (t_modes._timers.size() > 0)
		t_modes._timers[0].setStart();
	fftw_mpi_execute_dft_r2c(forward_plan, mesh, t_modes.fftw_modes);
	if (t_modes._timers.size() > 0)
		t_modes._timers[0].setDuration();
	if (t_norm)
		t_modes *= forward_dft_normalization;
	return;
}

void ScalarField::transformDFT(const bool t_norm) {
	if (_timers.size() > 0)
		_timers[0].setStart();
	fftw_mpi_execute_dft_r2c(forward_plan, mesh, fftw_modes);
	if (_timers.size() > 0)
		_timers[0].setDuration();
	if (t_norm)
		*this *= forward_dft_normalization;
	return;
}

void ScalarField::transformInverseDFT(ScalarField & t_mesh, const bool t_norm) const {
	if (t_mesh._timers.size() > 0)
		t_mesh._timers[0].setStart();
	fftw_mpi_execute_dft_c2r(backward_plan, fftw_modes, t_mesh.mesh);
	if (t_mesh._timers.size() > 0)
		t_mesh._timers[0].setDuration();
	if (t_norm)
		t_mesh *= backward_dft_normalization;
	return;
}

void ScalarField::transformInverseDFT(const bool t_norm) {
	if (_timers.size() > 0)
		_timers[0].setStart();
	fftw_mpi_execute_dft_c2r(backward_plan, fftw_modes, mesh);
	if (_timers.size() > 0)
		_timers[0].setDuration();
	if (t_norm)
		*this *= backward_dft_normalization;
	return;
}

/*
void ScalarField::conj(void) {
	auto imaginary_mode_view = _data | std::ranges::views::all | std::ranges::views::drop(1) | std::ranges::views::stride(2);
	std::uint64_t num_threads = getNumThreads();
	#pragma omp parallel
	{
		const std::uint64_t total_num_tasks = std::ranges::distance(imaginary_mode_view);
		const std::uint64_t remainder = total_num_tasks % num_threads;
		const std::uint64_t id = static_cast<std::uint64_t>(omp_get_thread_num());
		const std::uint64_t num_tasks = (id < remainder) ? total_num_tasks / num_threads + 1 : total_num_tasks / num_threads;
		const std::uint64_t start = (id < remainder) ? id * num_tasks : id * num_tasks + remainder;
		const std::uint64_t stop = std::min(start + num_tasks, total_num_tasks);
		auto thread_view = imaginary_mode_view | std::ranges::views::take(stop) | std::ranges::views::drop(start); 
		std::ranges::for_each(thread_view, [](auto & t_x){t_x = -t_x;});
	}
	return;
}
*/

void ScalarField::conj(void) {
	auto imaginary_mode_view = _data | std::ranges::views::all | std::ranges::views::drop(1) |
        std::views::filter([s = false](auto const&) mutable { return s = !s; });// | std::ranges::views::stride(2);
	std::uint64_t num_threads = getNumThreads();
	#pragma omp parallel
	{
		const std::uint64_t total_num_tasks = std::ranges::distance(imaginary_mode_view);
		const std::uint64_t remainder = total_num_tasks % num_threads;
		const std::uint64_t id = static_cast<std::uint64_t>(omp_get_thread_num());
		const std::uint64_t num_tasks = (id < remainder) ? total_num_tasks / num_threads + 1 : total_num_tasks / num_threads;
		const std::uint64_t start = (id < remainder) ? id * num_tasks : id * num_tasks + remainder;
		const std::uint64_t stop = std::min(start + num_tasks, total_num_tasks);
		auto thread_view = imaginary_mode_view | std::views::take(stop) | std::views::drop(start);
		std::ranges::for_each(thread_view, [](auto & t_x){t_x = -t_x;});
	}
	return;
}

void ScalarField::getGradientModes(const std::uint64_t t_dir, ScalarField & t_modes) const {
	if (t_modes._timers.size() > 1)
		t_modes._timers[1].setStart();
	#pragma omp parallel 
	{
		double wave_vector[3];
		std::uint64_t local_mode_index;
		#pragma omp for collapse(3)
		for (std::uint64_t i = local_start_x; i < local_end_x; i++) {
			for (std::uint64_t j = 0; j < num_mesh_1d; j++) {
				for (std::uint64_t k = 0; k <  num_modes_last_d; k++) {
					wave_vector[0] = wave_numbers[i];
					wave_vector[1] = wave_numbers[j];
					wave_vector[2] = wave_numbers[k];
					local_mode_index = getLocalModeIndex(i, j, k);
					t_modes.modes[local_mode_index] = 1i * wave_vector[t_dir] * modes[local_mode_index];
				}
			}
		}
	}
	if (t_modes._timers.size() > 1)
		t_modes._timers[1].setDuration();
	return;
}

void ScalarField::getGradient(const std::uint64_t t_dir, ScalarField & t_mesh) {
	if (!sizesAreConsistent(t_mesh))
		t_mesh = ScalarField(*this, false);
	transformDFT();
	getGradientModes(t_dir, t_mesh);
	t_mesh.transformInverseDFT();
	transformInverseDFT();
	return;
}

void ScalarField::getGradientModes(VectorField & t_mesh) const {
	if (!t_mesh.sizesAreConsistent(*this))
		t_mesh = VectorField(*this);
    for (std::uint64_t i = 0; i < 3; i++)
        getGradientModes(i, t_mesh[i]);
    return;
}

void ScalarField::getGradient(VectorField & t_mesh) {
	if (!t_mesh.sizesAreConsistent(*this))
		t_mesh = VectorField(*this);
    for (std::uint64_t i = 0; i < 3; i++)
        getGradient(i, t_mesh[i]);
    return;
}

void ScalarField::addGradientModes(const std::uint64_t t_dir, ScalarField & t_modes) const {
	if (t_modes._timers.size() > 1)
		t_modes._timers[1].setStart();
	#pragma omp parallel
	{
		double wave_vector[3];
		std::uint64_t local_mode_index;
		#pragma omp for collapse(3)
		for (std::uint64_t i = local_start_x; i < local_end_x; i++) {
			for (std::uint64_t j = 0; j < num_mesh_1d; j++) {
				for (std::uint64_t k = 0; k <  num_modes_last_d; k++) {
					wave_vector[0] = wave_numbers[i];
					wave_vector[1] = wave_numbers[j];
					wave_vector[2] = wave_numbers[k];
					local_mode_index = getLocalModeIndex(i, j, k);
					t_modes.modes[local_mode_index] += 1i * wave_vector[t_dir] * modes[local_mode_index];
				}
			}
		}
	}
	if (t_modes._timers.size() > 1)
		t_modes._timers[1].setDuration();
	return;
}

void ScalarField::addGradient(const std::uint64_t t_dir, ScalarField & t_mesh) {
	assert(sizesAreConsistent(t_mesh), "inconsistent scalar field mesh sizes for adding gradient");
	transformDFT();
	addGradientModes(t_dir, t_mesh);
	t_mesh.transformInverseDFT();
	transformInverseDFT();
	return;
}

void ScalarField::addGradient(VectorField & t_mesh) {
    for (std::uint64_t i = 0; i < 3; i++)
        addGradient(i, t_mesh[i]);
    return;
}

void ScalarField::getLaplacianModes(ScalarField & t_modes) const {
	if (t_modes._timers.size() > 1)
		t_modes._timers[1].setStart();
	#pragma omp parallel
	{
		double kx, ky, kz;
		std::uint64_t local_mode_index;
		#pragma omp for collapse(3)
		for (std::uint64_t i = local_start_x; i < local_end_x; i++) {
			for (std::uint64_t j = 0; j < num_mesh_1d; j++) {
				for (std::uint64_t k = 0; k <  num_modes_last_d; k++) {
					kx = wave_numbers[i];
					ky = wave_numbers[j];
					kz = wave_numbers[k];
					local_mode_index = getLocalModeIndex(i, j, k);
					t_modes.modes[local_mode_index] = -(kx * kx + ky * ky + kz * kz) * modes[local_mode_index];
				}
			}
		}
	}
	if (t_modes._timers.size() > 1)
		t_modes._timers[1].setDuration();
	return;
}

void ScalarField::getLaplacian(ScalarField & t_mesh) {
	if (!sizesAreConsistent(t_mesh))
		t_mesh = ScalarField(*this, false);
	transformDFT();
	getLaplacianModes(t_mesh);
	t_mesh.transformInverseDFT();
	transformInverseDFT();
	return;
}

void ScalarField::addLaplacianModes(ScalarField & t_modes) const {
	if (t_modes._timers.size() > 1)
		t_modes._timers[1].setStart();
	#pragma omp parallel
	{
		double kx, ky, kz;
		std::uint64_t local_mode_index;
		#pragma omp for collapse(3)
		for (std::uint64_t i = local_start_x; i < local_end_x; i++) {
			for (std::uint64_t j = 0; j < num_mesh_1d; j++) {
				for (std::uint64_t k = 0; k <  num_modes_last_d; k++) {
					kx = wave_numbers[i];
					ky = wave_numbers[j];
					kz = wave_numbers[k];
					local_mode_index = getLocalModeIndex(i, j, k);
					t_modes.modes[local_mode_index] += -(kx * kx + ky * ky + kz * kz) * modes[local_mode_index];
				}
			}
		}
	}
	if (t_modes._timers.size() > 1)
		t_modes._timers[1].setDuration();
	return;
}

void ScalarField::addLaplacian(ScalarField & t_mesh) {
	assert(sizesAreConsistent(t_mesh), "inconsistent scalar field mesh sizes for adding laplacian");
	transformDFT();
	addLaplacianModes(t_mesh);
	t_mesh.transformInverseDFT();
	transformInverseDFT();
	return;
}

void ScalarField::assign(const ScalarField & t_mesh, const double t_factor) {
	*this = t_mesh;
	if (t_factor != 1.)
		*this *= t_factor;
	return;
}

void ScalarField::add(const ScalarField & t_mesh, const double t_factor) {
	if (_timers.size() > 1)
		_timers[1].setStart();
	assert(sizesAreConsistent(t_mesh), "inconsistent " + label_string + " sizes for multiplying");
	if (t_factor == 0.)
		return;
	#pragma omp parallel
	{
		ConstTaskDistributor<double> tasks(t_mesh._data);
		std::vector<double>::iterator result = begin() + (tasks.begin - t_mesh.begin());
		if (t_factor != 1.)
			std::transform(tasks.begin, tasks.end, result, result, [t_factor](auto t_m2, auto t_m1){return t_m1 + t_factor * t_m2;});
		else
			std::transform(tasks.begin, tasks.end, result, result, [](auto t_m2, auto t_m1){return t_m1 + t_m2;});
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	return;
}

void ScalarField::assignProduct(const ScalarField & t_mesh_1, const ScalarField & t_mesh_2, const double t_factor) {
	if (_timers.size() > 1)
		_timers[1].setStart();
	assert(sizesAreConsistent(t_mesh_1), "inconsistent scalar field mesh sizes for assigning product");
	assert(sizesAreConsistent(t_mesh_2), "inconsistent scalar field mesh sizes for assigning product");
	if (t_factor == 0.) {
		*this *= 0.;
		return;
	}
	#pragma omp parallel
	{
		ConstTaskDistributor<double> tasks(t_mesh_1._data);
		std::vector<double>::const_iterator start2 = t_mesh_2.begin() + (tasks.begin - t_mesh_1.begin());
		std::vector<double>::iterator result = begin() + (tasks.begin - t_mesh_1.begin());
		if (t_factor != 1.)
			std::transform(tasks.begin, tasks.end, start2, result, [t_factor](auto t_m2, auto t_m1){return t_factor * t_m1 * t_m2;});
		else
			std::transform(tasks.begin, tasks.end, start2, result, [](auto t_m2, auto t_m1){return t_m1 * t_m2;});
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	return;
}

void ScalarField::addProduct(const ScalarField & t_mesh_1, const ScalarField & t_mesh_2, const double t_factor) {
	if (_timers.size() > 1)
		_timers[1].setStart();
	assert(sizesAreConsistent(t_mesh_1), "inconsistent scalar field mesh sizes for adding product");
	assert(sizesAreConsistent(t_mesh_2), "inconsistent scalar field mesh sizes for adding product");
	if (t_factor == 0.)
		return;
	#pragma omp parallel
	{
		ConstTaskDistributor<double> tasks(t_mesh_1._data);
		std::vector<double>::const_iterator start2 = t_mesh_2.begin() + (tasks.begin - t_mesh_1.begin());
		std::vector<double>::iterator result = begin() + (tasks.begin - t_mesh_1.begin());
		if (t_factor != 1.) {
			while (tasks.begin != tasks.end) {
				*result += t_factor * (*tasks.begin) * (*start2);
				++start2;
				++result;
				++tasks.begin;
			}
		}
		else {
			while (tasks.begin != tasks.end) {
				*result += (*tasks.begin) * (*start2);
				++start2;
				++result;
				++tasks.begin;
			}
		}    
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	return;   
}

void ScalarField::assignDotProduct(const VectorField & t_mesh_1, const VectorField & t_mesh_2, const double t_factor) {
	assert(t_mesh_2.sizesAreConsistent(t_mesh_1), "inconsistent vector field mesh sizes for assigning dot product");
	assignProduct(t_mesh_1.components[0], t_mesh_2.components[0], t_factor);
    for (std::uint64_t i = 1; i < 3; i++) {
		addProduct(t_mesh_1.components[i], t_mesh_2.components[i], t_factor);
    }
    return;   
}

void ScalarField::addDotProduct(const VectorField & t_mesh_1, const VectorField & t_mesh_2, const double t_factor) {
	for (std::uint64_t d = 0; d < 3; d++)
		addProduct(t_mesh_1.components[d], t_mesh_2.components[d], t_factor);
	return;   
}

void ScalarField::add(const double t_x) {
	if (_timers.size() > 1)
		_timers[1].setStart();
	if (t_x == 0.)
		return;
	#pragma omp parallel
	{
		TaskDistributor<double> tasks(_data);
		std::transform(tasks.begin, tasks.end, tasks.begin, [t_x](auto t_m){return t_m + t_x;});
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	return;
}

void ScalarField::subtract(const ScalarField & t_mesh, const double t_factor) {
	if (_timers.size() > 1)
		_timers[1].setStart();
	assert(sizesAreConsistent(t_mesh), "inconsistent scalar field mesh sizes for subtracting");
	#pragma omp parallel
	{
		ConstTaskDistributor<double> tasks(t_mesh._data);
		std::vector<double>::iterator result = begin() + (tasks.begin - t_mesh.begin());
		if (t_factor != 1.)
			std::transform(tasks.begin, tasks.end, result, result, [t_factor](auto t_m2, auto t_m1){return t_m1 - t_factor * t_m2;});
		else
			std::transform(tasks.begin, tasks.end, result, result, std::minus<double>());
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	return;
}

void ScalarField::multiply(const ScalarField & t_mesh, const double t_factor) {
	if (_timers.size() > 1)
		_timers[1].setStart();
	assert(sizesAreConsistent(t_mesh), "inconsistent scalar field mesh sizes for multiplying");
	if (t_factor == 0.) {
		*this *= 0;
		return;
	}
	#pragma omp parallel
	{
		ConstTaskDistributor<double> tasks(t_mesh._data);
		std::vector<double>::iterator result = begin() + (tasks.begin - t_mesh.begin());
		if (t_factor != 1.) 
			std::transform(tasks.begin, tasks.end, result, result, [t_factor](auto t_m2, auto t_m1){return t_factor * t_m1 * t_m2;});
		else
			std::transform(tasks.begin, tasks.end, result, result, std::multiplies<double>());
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	return;
}

void ScalarField::multiply(const double t_x) {
	if (_timers.size() > 1)
		_timers[1].setStart();
	#pragma omp parallel
	{	
		TaskDistributor<double> tasks(_data);
		std::for_each(tasks.begin, tasks.end, [t_x](auto & t_m){t_m *= t_x;});
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	return;
}

void ScalarField::divide(const ScalarField & t_mesh, const double t_factor) {
	if (_timers.size() > 1)
		_timers[1].setStart();
	assert(sizesAreConsistent(t_mesh), "inconsistent scalar field mesh sizes for dividing");
	if (t_factor == 0.) {
		*this *= 0.;
		return;
	}
	#pragma omp parallel
	{
		ConstTaskDistributor<double> tasks(t_mesh._data);
		std::vector<double>::iterator result = begin() + (tasks.begin - t_mesh.begin());
		if (t_factor != 1.)
			std::transform(tasks.begin, tasks.end, result, result, [t_factor](auto t_m2, auto t_m1){return t_factor * t_m1 / t_m2;});
		else
			std::transform(tasks.begin, tasks.end, result, result, [](auto t_m2, auto t_m1){return t_m1 / t_m2;});
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	return;
}

void ScalarField::zeroPadding(void) {
	if (_timers.size() > 1)
		_timers[1].setStart();
	#pragma omp parallel for collapse(3) 
	for (std::uint64_t i = local_start_x; i < local_end_x; i++) {
		for (std::uint64_t j = 0; j < num_mesh_1d; j++) {
			for (std::uint64_t k = two_num_modes_last_d - num_padding_1d; k < two_num_modes_last_d; k++)
				_data[getLocalMeshIndex(i, j, k)] = 0.;
		}
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	return;
}

const double ScalarField::sum(void) {
	zeroPadding();
	if (_timers.size() > 1)
		_timers[1].setStart();
	double sum = 0.;
	#pragma omp parallel
	{
		TaskDistributor<double> tasks(_data);
		double thread_sum = std::accumulate(tasks.begin, tasks.end, 0.);
		#pragma omp atomic
		sum += thread_sum;
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	if (_timers.size() > 2)
		_timers[2].setStart();
	MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (_timers.size() > 2)
		_timers[2].setDuration();
	return sum;
}

const double ScalarField::sumProduct(const ScalarField & t_mesh) {
	assert(sizesAreConsistent(t_mesh), "inconsistent scalar field mesh sizes for summing product");
	zeroPadding();
	if (_timers.size() > 1)
		_timers[1].setStart();
	double sum = 0.;
	#pragma omp parallel
	{
		ConstTaskDistributor<double> tasks(t_mesh._data);
		double thread_sum = 0.;
		std::vector<double>::iterator start2 = begin() + (tasks.begin - t_mesh.begin());
		while (tasks.begin != tasks.end) {
			thread_sum += (*tasks.begin) * (*start2);
			++start2;
			++tasks.begin;
		}
		#pragma omp atomic
		sum += thread_sum;
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	if (_timers.size() > 2)
		_timers[2].setStart();
	MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if (_timers.size() > 2)
		_timers[2].setDuration();
	return sum;
}


void ScalarField::enforceHermiticity(void) {
    std::vector<std::complex<double>> front_slab(num_mesh_1d * num_mesh_1d, 0.);
    #pragma omp parallel
    {
        std::uint64_t mode_index;
        std::uint64_t slab_index;
        #pragma omp for collapse(2)
        for (std::uint64_t i = local_start_x; i < local_end_x; i++) {
            for (std::uint64_t j = 0; j < num_mesh_1d; j++) {
                mode_index = getLocalModeIndex(i, j, 0);
                slab_index = j + num_mesh_1d * i;
                front_slab[slab_index] = modes[mode_index];
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, front_slab.data(), front_slab.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    #pragma omp parallel
    {
        std::uint64_t base_index;
        std::uint64_t conjugate_index;
        double base_amplitude;
        double conjugate_amplitude;
        double amplitude;
        std::uint64_t mode_index;
        std::uint64_t slab_index;
        #pragma omp for
        for (std::uint64_t i = num_modes_last_d; i < num_mesh_1d; i++) {
            base_index = num_mesh_1d - i;
            conjugate_index = i;
            base_amplitude = std::abs(front_slab[base_index]);
            conjugate_amplitude = std::abs(front_slab[conjugate_index]);
            amplitude = sqrt((base_amplitude * base_amplitude + conjugate_amplitude * conjugate_amplitude) / 2.);
            base_amplitude = amplitude != 0. ? base_amplitude : 1.;
            front_slab[base_index] *= amplitude / base_amplitude;
            front_slab[conjugate_index] = std::conj(front_slab[base_index]);
            base_index = num_mesh_1d * (num_mesh_1d - i);
            conjugate_index = num_mesh_1d * i;
            base_amplitude = std::abs(front_slab[base_index]);
            conjugate_amplitude = std::abs(front_slab[conjugate_index]);
            amplitude = sqrt((base_amplitude * base_amplitude + conjugate_amplitude * conjugate_amplitude) / 2.);
            base_amplitude = amplitude != 0. ? base_amplitude : 1.;
            front_slab[base_index] *= amplitude / base_amplitude;
            front_slab[conjugate_index] = std::conj(front_slab[base_index]);
        }
        #pragma omp for collapse(2)
        for (std::uint64_t i = num_modes_last_d; i < num_mesh_1d; i++) {
            for (std::uint64_t j = num_modes_last_d; j < num_mesh_1d; j++) {
                base_index = (num_mesh_1d - j) + num_mesh_1d * i;
                conjugate_index = j + num_mesh_1d * (num_mesh_1d - i) ;
                base_amplitude = std::abs(front_slab[base_index]);
                conjugate_amplitude = std::abs(front_slab[conjugate_index]);
                amplitude = sqrt((base_amplitude * base_amplitude + conjugate_amplitude * conjugate_amplitude) / 2.);
                base_amplitude = amplitude != 0. ? base_amplitude : 1.;
                front_slab[base_index] *= amplitude / base_amplitude;
                front_slab[conjugate_index] = std::conj(front_slab[base_index]);
                base_index = (num_mesh_1d - j) + num_mesh_1d * (num_mesh_1d - i);
                conjugate_index = j + num_mesh_1d * i ;
                base_amplitude = std::abs(front_slab[base_index]);
                conjugate_amplitude = std::abs(front_slab[conjugate_index]);
                amplitude = sqrt((base_amplitude * base_amplitude + conjugate_amplitude * conjugate_amplitude) / 2.);
                base_amplitude = amplitude != 0. ? base_amplitude : 1.;
                front_slab[base_index] *= amplitude / base_amplitude;
                front_slab[conjugate_index] = std::conj(front_slab[base_index]);
            }
        }
        #pragma omp for collapse(2)
        for (std::uint64_t i = local_start_x; i < local_end_x; i++) {
            for (std::uint64_t j = 0; j < num_mesh_1d; j++) {
                mode_index = getLocalModeIndex(i, j, 0);
                slab_index = j + num_mesh_1d * i;
                modes[mode_index] = front_slab[slab_index];
            }
        }
    }
    return;
}

const ScalarField & ScalarField::operator=(const ScalarField & t_mesh) {
	if (_timers.size() > 1)
		_timers[1].setStart();
	#pragma omp parallel
	{
		ConstTaskDistributor<double> tasks(t_mesh._data);
		std::vector<double>::iterator result = begin() + (tasks.begin - t_mesh._data.begin());
		std::copy(tasks.begin, tasks.end, result);
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	return *this;
}

const ScalarField & ScalarField::operator=(const double t_x) {
	if (_timers.size() > 1)
		_timers[1].setStart();
	#pragma omp parallel
	{
		TaskDistributor<double> tasks(_data);
		std::fill(tasks.begin, tasks.end, t_x);
	}
	if (_timers.size() > 1)
		_timers[1].setDuration();
	return *this;
}

const ScalarField operator-(ScalarField & t_mesh) {
    return operate(t_mesh, std::negate<double>());
}

const ScalarField operator+(ScalarField & t_mesh, double t_x) {
    return operate(t_mesh, [t_x](auto t){return t + t_x;});
}

const ScalarField operator+(double t_x, ScalarField & t_mesh) {
    return operator+(t_mesh, t_x);
}

const ScalarField operator+(ScalarField & t_mesh_1, ScalarField & t_mesh_2) {
    return operate(t_mesh_1, t_mesh_2, std::plus<double>());
}

const ScalarField operator-(ScalarField & t_mesh, double t_x) {
    return operate(t_mesh, [t_x](auto t){return t - t_x;});
}

const ScalarField operator-(double t_x, ScalarField & t_mesh) {
    return operate(t_mesh, [t_x](auto t){return t_x - t;});
}

const ScalarField operator-(ScalarField & t_mesh_1, ScalarField & t_mesh_2) {
    return operate(t_mesh_1, t_mesh_2, std::minus<double>());
}

const ScalarField operator*(ScalarField & t_mesh, double t_x) {
    return operate(t_mesh, [t_x](auto t){return t * t_x;});
}

const ScalarField operator*(double t_x, ScalarField & t_mesh) {
    return operator*(t_mesh, t_x);
}

const ScalarField operator*(ScalarField & t_mesh_1, ScalarField & t_mesh_2) {
    return operate(t_mesh_1, t_mesh_2, std::multiplies<double>());
}

const ScalarField operator/(ScalarField & t_mesh, double t_x) {
    return operate(t_mesh, [t_x](auto t){return t / t_x;});
}

const ScalarField operator/(double t_x, ScalarField & t_mesh) {
    return operate(t_mesh, [t_x](auto t){return t_x / t;});
}

const ScalarField operator/(ScalarField & t_mesh_1, ScalarField & t_mesh_2) {
    return operate(t_mesh_1, t_mesh_2, std::divides<double>());
}

const ScalarField & ScalarField::operator+=(const ScalarField & t_mesh) {
    add(t_mesh);
    return *this;
}

const ScalarField & ScalarField::operator+=(const double t_x) {
    add(t_x);
    return *this;
}

const ScalarField & ScalarField::operator-=(const ScalarField & t_mesh) {
    subtract(t_mesh);
    return *this;
}

const ScalarField & ScalarField::operator-=(const double t_x) {
    add(-t_x);
    return *this;
}

const ScalarField & ScalarField::operator*=(const ScalarField & t_mesh) {
    multiply(t_mesh);
    return *this;
}

const ScalarField & ScalarField::operator*=(const double t_x) {
    multiply(t_x);
    return *this;
}

const ScalarField & ScalarField::operator/=(const ScalarField & t_mesh) {
    divide(t_mesh);
    return *this;
}

const ScalarField & ScalarField::operator/=(const double t_x) {
	const double inv_x = 1. / t_x;
    multiply(inv_x);
    return *this;
}
