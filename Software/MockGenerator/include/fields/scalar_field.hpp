//************************
//*
//*
//*
//*
//************************

#ifndef SCALAR_FIELD_H
#define SCALAR_FIELD_H

#include<fields/field.hpp>

using namespace std::complex_literals;
class VectorField;

class ScalarField : public Field {

    public :

    ScalarField(void) {
		exception_prefix = "Error::ScalarField";
		label_string = "scalar field";
	}

    ScalarField(const std::uint64_t t_num_mesh_1d, const double t_box_length) {
        initialize(t_num_mesh_1d, t_box_length);
    }

    ScalarField(const ScalarField & t_mesh, const bool t_copy_data) : ScalarField(t_mesh.num_mesh_1d, t_mesh.box_length) {
        if (t_copy_data) {
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
        }
    }

    ScalarField(const ScalarField & t_mesh) : ScalarField(t_mesh, true) {}

    void initialize(const std::uint64_t t_num_mesh_1d, const double t_box_length);

    std::vector<double> _data;
    double * mesh;
    std::complex<double> * modes;
    fftw_complex * fftw_modes;
    fftw_plan forward_plan;
    fftw_plan backward_plan;

    double & operator[](std::uint64_t t_i);
    double * data(void);
    const std::vector<double>::iterator begin(void);
    const std::vector<double>::iterator end(void);
	const std::vector<double>::const_iterator begin(void) const;
    const std::uint64_t size(void) const;

    const double getField(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const;
    const std::complex<double> getMode(const std::uint64_t t_i, const std::uint64_t t_j, const std::uint64_t t_k) const;

    void transformDFT(ScalarField & t_modes, const bool t_norm = true) const;
    void transformDFT(const bool t_norm = true);
    void transformInverseDFT(ScalarField & t_mesh, const bool t_norm = true) const;
    void transformInverseDFT(const bool t_norm = true);
	void conj(void);

    void getGradientModes(const std::uint64_t t_dir, ScalarField & t_modes) const;
    void getGradient(const std::uint64_t t_dir, ScalarField & t_mesh);
	void getGradientModes(VectorField & t_mesh) const;
    void getGradient(VectorField & t_mesh);
    void addGradientModes(const std::uint64_t t_dir, ScalarField & t_modes) const;
    void addGradient(const std::uint64_t t_dir, ScalarField & t_mesh);
    void addGradient(VectorField & t_mesh);
    void getLaplacianModes(ScalarField & t_modes) const;
    void getLaplacian(ScalarField & t_mesh);
    void addLaplacianModes(ScalarField & t_modes) const;
    void addLaplacian(ScalarField & t_mesh);

    void assign(const ScalarField & t_mesh, const double t_factor = 1.);
    void add(const ScalarField & t_mesh, const double t_factor = 1.);
    void assignProduct(const ScalarField & t_mesh_1, const ScalarField & t_mesh_2, const double t_factor = 1.);
    void addProduct(const ScalarField & t_mesh_1, const ScalarField & t_mesh_2, const double t_factor = 1.);
    void add(const double t_x);
    void subtract(const ScalarField & t_mesh, const double t_factor = 1.);
    void multiply(const ScalarField & t_mesh, const double t_factor = 1.);
    void multiply(const double t_x);
    void divide(const ScalarField & t_mesh, const double t_factor = 1.);
    void assignDotProduct(const VectorField & t_mesh_1, const VectorField & t_mesh_2, const double t_factor = 1.);
	void addDotProduct(const VectorField & t_mesh_1, const VectorField & t_mesh_2, const double t_factor = 1.);

	void zeroPadding(void);
    const double sum(void);
    const double sumProduct(const ScalarField & t_mesh);
    void enforceHermiticity(void);

    const ScalarField & operator=(const ScalarField & t_mesh);
    const ScalarField & operator=(const double t_x);
    const ScalarField & operator+=(const ScalarField & t_mesh);
    const ScalarField & operator+=(const double t_x);
    const ScalarField & operator-=(const ScalarField & t_mesh);
    const ScalarField & operator-=(const double t_x);
    const ScalarField & operator*=(const ScalarField & t_mesh);
    const ScalarField & operator*=(const double t_x);
    const ScalarField & operator/=(const ScalarField & t_mesh);
    const ScalarField & operator/=(const double t_x);

    template<typename functional>
    const double sum(const functional & f) {
		zeroPadding();
		if (_timers.size() > 1)
			_timers[1].setStart();
        double sum = 0.;
        #pragma omp parallel
        {
            TaskDistributor<double> tasks(_data);
            double thread_sum = std::accumulate(tasks.begin, tasks.end, 0., [&f](const auto t_x1, const auto t_x2){return t_x1 + f(t_x2);});
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

    template <typename functional>
    void applyFunction(const functional & f) {
		if (_timers.size() > 1)
			_timers[1].setStart();
        #pragma omp parallel
        {
            TaskDistributor<double> tasks(_data);
            std::transform(tasks.begin, tasks.end, tasks.begin, f);
        }
		if (_timers.size() > 1)
			_timers[1].setDuration();
		return;
    }

};

template<typename functional>
const ScalarField operate(ScalarField & t_mesh, const functional & f) {
    ScalarField mesh(t_mesh);
	if (t_mesh._timers.size() > 1)
		t_mesh._timers[1].setStart();
    #pragma omp parallel
    {
        ConstTaskDistributor<double> tasks(t_mesh._data);
        std::vector<double>::iterator result = mesh.begin() + (tasks.begin - t_mesh.begin());
        std::transform(tasks.begin, tasks.end, result, f);
    }
	if (t_mesh._timers.size() > 1)
		t_mesh._timers[1].setDuration();
    return mesh;
}

template<typename functional>
const ScalarField operate(ScalarField & t_mesh_1, const ScalarField & t_mesh_2, const functional & f) {
    t_mesh_1.assert(t_mesh_1.sizesAreConsistent(t_mesh_2), "inconsistent scalar field mesh sizes for function operation");
    ScalarField mesh(t_mesh_1);
	if (t_mesh_1._timers.size() > 1)
		t_mesh_1._timers[1].setStart();
    #pragma omp parallel
    {
        TaskDistributor<double> tasks(t_mesh_1._data);
        std::vector<double>::const_iterator start2 = t_mesh_2.begin() + (tasks.begin - t_mesh_1.begin());
        std::vector<double>::iterator result = mesh.begin() + (tasks.begin - t_mesh_1.begin());
        std::transform(tasks.begin, tasks.end, start2, result, f);
    }
	if (t_mesh_1._timers.size() > 1)
		t_mesh_1._timers[1].setDuration();
    return mesh;
}

#endif // *** SCALAR_FIELD_MESH_H *** //
