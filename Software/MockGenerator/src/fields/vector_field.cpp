#include<fields/vector_field.hpp>

void VectorField::initialize(const std::uint64_t t_num_mesh_1d, const double t_box_length) {
	try {
		Field::initialize(t_num_mesh_1d, t_box_length);
		components.resize(3);
		for (std::uint64_t i = 0; i < 3; i++)
			components[i].initialize(t_num_mesh_1d, t_box_length);
	}
	catch (std::runtime_error & t_error) {
		if (getNodeID() == 0)
			std::cerr << t_error.what();
		assert(false, "failed to allocate " + label_string + " mesh");
	}
	return;
}

ScalarField & VectorField::operator[](const std::uint64_t t_i) {
	return components[t_i];
}

void VectorField::transformDFT(VectorField & t_modes, const bool t_norm) const {
	for (std::uint64_t d = 0; d < 3; d++)
		components[d].transformDFT(t_modes[d], t_norm);
	return;
}

void VectorField::transformDFT(const bool t_norm) {
	for (auto & c : components)
		c.transformDFT(t_norm);
	return;
}

void VectorField::transformInverseDFT(VectorField & t_mesh, const bool t_norm) const {
	for (std::uint64_t d = 0; d < 3; d++)
		components[d].transformInverseDFT(t_mesh[d], t_norm);
	return;
}

void VectorField::transformInverseDFT(const bool t_norm) {
	for (auto & c : components)
		c.transformInverseDFT(t_norm);
	return;
}

void VectorField::getCurlModes(VectorField & t_modes) const {
	if (t_modes[0]._timers.size() > 1)
		t_modes[0]._timers[1].setStart();
	#pragma omp parallel
	{
		ScalarField * c = &t_modes[0];
		double wave_vector[3];
		std::uint64_t local_mode_index;
		#pragma omp for collapse(3)
		for (std::uint64_t i = c->local_start_x; i < c->local_end_x; i++) {
			for (std::uint64_t j = 0; j < c->num_mesh_1d; j++) {
				for (std::uint64_t k = 0; k <  num_modes_last_d; k++) {
					wave_vector[0] = c->wave_numbers[i];
					wave_vector[1] = c->wave_numbers[j];
					wave_vector[2] = c->wave_numbers[k];
					local_mode_index = c->getLocalModeIndex(i, j, k);
					for (std::uint64_t d = 0; d < 3; d++) {
						t_modes[d].modes[local_mode_index] = 1i * wave_vector[(d + 1) % 3] * components[(d + 2) % 3].modes[local_mode_index];
						t_modes[d].modes[local_mode_index] -= 1i * wave_vector[(d + 2) % 3] * components[(d + 1) % 3].modes[local_mode_index];
					}
				}
			}
		}
	}
	if (t_modes[0]._timers.size() > 1)
		t_modes[0]._timers[1].setDuration();
	return;
}

void VectorField::getCurl(VectorField & t_mesh) {
	transformDFT();
	getCurlModes(t_mesh);
	t_mesh.transformInverseDFT();
	transformInverseDFT();
	return;
}

void VectorField::addCurlModes(VectorField & t_modes) const {
	if (t_modes[0]._timers.size() > 1)
		t_modes[0]._timers[1].setStart();
	#pragma omp parallel
	{
		ScalarField * c = &t_modes[0];
		double wave_vector[3];
		std::uint64_t local_mode_index;
		#pragma omp for collapse(3)
		for (std::uint64_t i = c->local_start_x; i < c->local_end_x; i++) {
			for (std::uint64_t j = 0; j < c->num_mesh_1d; j++) {
				for (std::uint64_t k = 0; k <  num_modes_last_d; k++) {
					wave_vector[0] = c->wave_numbers[i];
					wave_vector[1] = c->wave_numbers[j];
					wave_vector[2] = c->wave_numbers[k];
					local_mode_index = c->getLocalModeIndex(i, j, k);
					for (std::uint64_t d = 0; d < 3; d++) {
						t_modes[d].modes[local_mode_index] += 1i * wave_vector[(d + 1) % 3] * components[(d + 2) % 3].modes[local_mode_index];
						t_modes[d].modes[local_mode_index] -= 1i * wave_vector[(d + 2) % 3] * components[(d + 1) % 3].modes[local_mode_index];
					}
				}
			}
		}
	}
	if (t_modes[0]._timers.size() > 1)
		t_modes[0]._timers[1].setDuration();
	return;
}

void VectorField::addCurl(VectorField & t_mesh) {
	transformDFT();
	addCurlModes(t_mesh);
	t_mesh.transformInverseDFT();
	transformInverseDFT();
	return;
}

void VectorField::getDivergenceModes(ScalarField & t_modes) const {
	if (t_modes._timers.size() > 1)
		t_modes._timers[1].setStart();
	#pragma omp parallel
	{
		ScalarField * c = &t_modes;
		double wave_vector[3];
		std::uint64_t local_mode_index;
		#pragma omp for collapse(3)
		for (std::uint64_t i = c->local_start_x; i < c->local_end_x; i++) {
			for (std::uint64_t j = 0; j < c->num_mesh_1d; j++) {
				for (std::uint64_t k = 0; k <  num_modes_last_d; k++) {
					wave_vector[0] = c->wave_numbers[i];
					wave_vector[1] = c->wave_numbers[j];
					wave_vector[2] = c->wave_numbers[k];
					local_mode_index = c->getLocalModeIndex(i, j, k);
					t_modes.modes[local_mode_index] = 0.;
					for (std::uint64_t d = 0; d < 3; d++)
						t_modes.modes[local_mode_index] += 1i * wave_vector[d] * components[d].modes[local_mode_index];
				}
			}
		}
	}
	if (t_modes._timers.size() > 1)
		t_modes._timers[1].setDuration();
	return;
}

void VectorField::getDivergence(ScalarField & t_mesh) {
	transformDFT();
	getDivergenceModes(t_mesh);
	t_mesh.transformInverseDFT();
	transformInverseDFT();
	return;
}

void VectorField::addDivergenceModes(ScalarField & t_modes) const {
	if (t_modes._timers.size() > 1)
		t_modes._timers[1].setStart();
	#pragma omp parallel
	{
		ScalarField * c = &t_modes;
		double wave_vector[3];
		std::uint64_t local_mode_index;
		#pragma omp for collapse(3)
		for (std::uint64_t i = c->local_start_x; i < c->local_end_x; i++) {
			for (std::uint64_t j = 0; j < c->num_mesh_1d; j++) {
				for (std::uint64_t k = 0; k <  num_modes_last_d; k++) {
					wave_vector[0] = c->wave_numbers[i];
					wave_vector[1] = c->wave_numbers[j];
					wave_vector[2] = c->wave_numbers[k];
					local_mode_index = c->getLocalModeIndex(i, j, k);
					for (std::uint64_t d = 0; d < 3; d++)
						t_modes.modes[local_mode_index] += 1i * wave_vector[d] * components[d].modes[local_mode_index];
				}
			}
		}
	}
	if (t_modes._timers.size() > 1)
		t_modes._timers[1].setDuration();
	return;
}

void VectorField::addDivergence(ScalarField & t_mesh) {
	transformDFT();
	t_mesh.transformDFT();
	addDivergenceModes(t_mesh);
	t_mesh.transformInverseDFT();
	transformInverseDFT();
	return;
}

void VectorField::getLaplacianModes(VectorField & t_mesh) const {
	for (std::uint64_t d = 0; d < 3; d++)
		components[d].getLaplacianModes(t_mesh[d]);
	return;
}

void VectorField::addLaplacianModes(VectorField & t_mesh) const {
	for (std::uint64_t d = 0; d < 3; d++)
		components[d].addLaplacianModes(t_mesh[d]);
	return;
}

void VectorField::getLaplacian(VectorField & t_mesh) {
	for (std::uint64_t d = 0; d < 3; d++)
		components[d].getLaplacian(t_mesh[d]);
	return;
}

void VectorField::addLaplacian(VectorField & t_mesh) {
	for (std::uint64_t d = 0; d < 3; d++)
		components[d].addLaplacian(t_mesh[d]);
	return;
}

void VectorField::assign(const VectorField & t_mesh, const double t_factor) {
	*this = t_mesh;
	if (t_factor != 1.)
		*this *= t_factor;
	return;
}

void VectorField::assign(const ScalarField & t_mesh, const double t_factor) {
    for (auto & c : components) 
		c.assign(t_mesh, t_factor);
	return;
}

void VectorField::add(const VectorField & t_mesh, const double t_factor) {
	for (std::uint64_t d = 0; d < 3; d++)
		components[d].add(t_mesh.components[d], t_factor);
	return;
}

void VectorField::add(const ScalarField & t_mesh, const double t_factor) {
    for (auto & c : components) 
		c.add(t_mesh, t_factor);
	return;
}

void VectorField::subtract(const VectorField & t_mesh, const double t_factor) {
	for (std::uint64_t d = 0; d < 3; d++)
		components[d].subtract(t_mesh.components[d], t_factor);
	return;
}

void VectorField::subtract(const ScalarField & t_mesh, const double t_factor) {
    for (auto & c : components) 
		c.subtract(t_mesh, t_factor);
	return;
}

void VectorField::multiply(const VectorField & t_mesh, const double t_factor) {
	for (std::uint64_t d = 0; d < 3; d++)
		components[d].multiply(t_mesh.components[d], t_factor);
	return;
}

void VectorField::multiply(const ScalarField & t_mesh, const double t_factor) {
    for (auto & c : components) 
		c.multiply(t_mesh, t_factor);
	return;
}

void VectorField::divide(const VectorField & t_mesh, const double t_factor) {
	for (std::uint64_t d = 0; d < 3; d++)
		components[d].divide(t_mesh.components[d], t_factor);
	return;
}

void VectorField::divide(const ScalarField & t_mesh, const double t_factor) {
    for (auto & c : components) 
		c.divide(t_mesh, t_factor);
	return;
}

void VectorField::assignCrossProduct(const VectorField & t_mesh_1, const VectorField & t_mesh_2, const double t_factor) {
	std::uint64_t i, j ,k;
    for (i = 0; i < 3; i++) {
		j = (i + 1) % 3;
		k = (i + 2) % 3;
		components[i].assignProduct(t_mesh_1.components[j], t_mesh_2.components[k], t_factor);
		components[i].addProduct(t_mesh_1.components[k], t_mesh_2.components[j], -t_factor);
    }
    return;   
}

void VectorField::addCrossProduct(const VectorField & t_mesh_1, const VectorField & t_mesh_2, const double t_factor) {
	std::uint64_t i, j ,k;
    for (i = 0; i < 3; i++) {
		j = (i + 1) % 3;
		k = (i + 2) % 3;
		components[i].addProduct(t_mesh_1.components[j], t_mesh_2.components[k], t_factor);
		components[i].addProduct(t_mesh_1.components[k], t_mesh_2.components[j], -t_factor);
    }
    return;   
}

void VectorField::assignProduct(const VectorField & t_field_1, const ScalarField & t_field_2, const double t_factor) {
    for (std::uint64_t d = 0; d < 3; d++) 
		components[d].assignProduct(t_field_1.components[d], t_field_2, t_factor);
	return;
}

void VectorField::assignProduct(const ScalarField & t_field_1, const VectorField & t_field_2, const double t_factor) {
	assignProduct(t_field_2, t_field_1, t_factor);
	return;
}

void VectorField::addProduct(const VectorField & t_field_1, const ScalarField & t_field_2, const double t_factor) {
    for (std::uint64_t d = 0; d < 3; d++) 
		components[d].addProduct(t_field_1.components[d], t_field_2, t_factor);
	return;
}

void VectorField::addProduct(const ScalarField & t_field_1, const VectorField & t_field_2, const double t_factor) {
	addProduct(t_field_2, t_field_1, t_factor);
	return;
}

const double VectorField::sumDotProduct(const VectorField & t_mesh) {
	assert(sizesAreConsistent(t_mesh), "inconsistent vector field mesh sizes for summing dot product");
	double sum = 0.;
	for (std::uint64_t d = 0; d < 3; d++)
		sum += components[d].sumProduct(t_mesh.components[d]);
	return sum;
}

const double VectorField::sumDotProduct(void) {
	double sum = 0.;
    for (auto & c : components) 
		sum += c.sum([](const auto t_x){return t_x * t_x;});
	return sum;
}

void VectorField::enforceHermiticity(void) {
	for (auto & c : components)
        c.enforceHermiticity();
	return;
}

void VectorField::conj(void) {
	for (auto & c : components)
		c.conj();
	return;
}

const VectorField operator-(VectorField & t_mesh) {
    return operate(t_mesh, std::negate<double>());
}

const VectorField operator+(VectorField & t_mesh, double t_x) {
    return operate(t_mesh, [t_x](auto t){return t + t_x;});
}

const VectorField operator+(double t_x, VectorField & t_mesh) {
    return operator+(t_mesh, t_x);
}

const VectorField operator+(VectorField & t_mesh_1, VectorField & t_mesh_2) {
    return operate(t_mesh_1, t_mesh_2, std::plus<double>());
}

const VectorField operator-(VectorField & t_mesh, double t_x) {
    return operate(t_mesh, [t_x](auto t){return t - t_x;});
}

const VectorField operator-(double t_x, VectorField & t_mesh) {
    return operate(t_mesh, [t_x](auto t){return t_x - t;});
}

const VectorField operator-(VectorField & t_mesh_1, VectorField & t_mesh_2) {
    return operate(t_mesh_1, t_mesh_2, std::minus<double>());
}

const VectorField operator*(VectorField & t_mesh, double t_x) {
    return operate(t_mesh, [t_x](auto t){return t * t_x;});
}

const VectorField operator*(double t_x, VectorField & t_mesh) {
    return operator*(t_mesh, t_x);
}

const VectorField operator*(VectorField & t_mesh_1, VectorField & t_mesh_2) {
    return operate(t_mesh_1, t_mesh_2, std::multiplies<double>());
}

const VectorField operator/(VectorField & t_mesh, double t_x) {
    return operate(t_mesh, [t_x](auto t){return t / t_x;});
}

const VectorField operator/(double t_x, VectorField & t_mesh) {
    return operate(t_mesh, [t_x](auto t){return t_x / t;});
}

const VectorField operator/(VectorField & t_mesh_1, VectorField & t_mesh_2) {
    return operate(t_mesh_1, t_mesh_2, std::divides<double>());
}

const VectorField & VectorField::operator=(const VectorField & t_mesh) {
    for (std::uint64_t d = 0; d < 3; d++)
		components[d] = t_mesh.components[d];
	return *this;
}

const VectorField & VectorField::operator=(const ScalarField & t_mesh) {
    for (auto & c : components)
		c = t_mesh;
	return *this;
}

const VectorField & VectorField::operator=(const double t_x) {
    for (auto & c : components)
		c = t_x;
	return *this;
}

const VectorField & VectorField::operator+=(const VectorField & t_mesh) {
    for (std::uint64_t d = 0; d < 3; d++)
        components[d] += t_mesh.components[d];
    return *this;
}

const VectorField & VectorField::operator+=(const ScalarField & t_mesh) {
    for (auto & c : components)
        c += t_mesh;
    return *this;
}

const VectorField & VectorField::operator+=(const double t_x) {
    for (auto & c : components)
        c += t_x;
    return *this;
}

const VectorField & VectorField::operator-=(const VectorField & t_mesh) {
    for (std::uint64_t d = 0; d < 3; d++)
        components[d] -= t_mesh.components[d];
    return *this;
}

const VectorField & VectorField::operator-=(const ScalarField & t_mesh) {
    for (auto & c : components)
        c -= t_mesh;
    return *this;
}

const VectorField & VectorField::operator-=(const double t_x) {
    for (auto & c : components)
        c -= t_x;
    return *this;
}

const VectorField & VectorField::operator*=(const VectorField & t_mesh) {
    for (std::uint64_t d = 0; d < 3; d++)
        components[d] *= t_mesh.components[d];
    return *this;
}

const VectorField & VectorField::operator*=(const ScalarField & t_mesh) {
    for (auto & c : components)
        c *= t_mesh;
    return *this;
}

const VectorField & VectorField::operator*=(const double t_x) {
    for (auto & c : components)
        c *= t_x;
    return *this;
}

const VectorField & VectorField::operator/=(const VectorField & t_mesh) {
    for (std::uint64_t d = 0; d < 3; d++)
        components[d] /= t_mesh.components[d];
    return *this;
}

const VectorField & VectorField::operator/=(const ScalarField & t_mesh) {
    for (auto & c : components)
        c /= t_mesh;
    return *this;
}

const VectorField & VectorField::operator/=(const double t_x) {
    for (auto & c : components)
        c /= t_x;
    return *this;
}
