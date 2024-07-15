//************************
//*
//*
//*
//*
//************************

#ifndef VECTOR_FIELD_H
#define VECTOR_FIELD_H

#include<fields/scalar_field.hpp>

class VectorField : public Field {

    public:

    VectorField(void){
		exception_prefix = "Error::VectorField, ";
		label_string = "vector field";
	}

    VectorField(const std::uint64_t t_num_mesh_1d, const double t_box_length) : VectorField() {
        initialize(t_num_mesh_1d, t_box_length);
    }

    VectorField(const VectorField & t_field, const bool t_copy_data) : VectorField(t_field.num_mesh_1d, t_field.box_length) {
        for (std::uint64_t i = 0; i < 3; i++)
            components.push_back(ScalarField(t_field.components[i], t_copy_data));
    }

    VectorField(const VectorField & t_field) : VectorField(t_field, true) {}

    VectorField(const ScalarField & t_field) : VectorField(t_field.num_mesh_1d, t_field.box_length) {}

    void initialize(const std::uint64_t t_num_mesh_1d, const double t_box_length);

    std::vector<ScalarField> components;

    ScalarField & operator[](const std::uint64_t t_i);
    void transformDFT(VectorField & t_modes, const bool t_norm = true) const;
    void transformDFT(const bool t_norm = true);
    void transformInverseDFT(VectorField & t_field, const bool t_norm = true) const;
    void transformInverseDFT(const bool t_norm = true);

    template <typename functional>
    void applyFunction(const functional & f) {
        for (std::uint64_t i = 0; i < 3; i++)
            components[i].applyFunction(f);
        return;
    }

    void getCurlModes(VectorField & t_modes) const;
    void getCurl(VectorField & t_field);
    void addCurlModes(VectorField & t_modes) const;
    void addCurl(VectorField & t_field);
    void getDivergenceModes(ScalarField & t_modes) const;
    void getDivergence(ScalarField & t_field);
    void addDivergenceModes(ScalarField & t_modes) const;
    void addDivergence(ScalarField & t_field);
	void getLaplacianModes(VectorField & t_mesh) const;
	void addLaplacianModes(VectorField & t_mesh) const;
    void getLaplacian(VectorField & t_field);
    void addLaplacian(VectorField & t_field);

    void assign(const VectorField & t_field, const double t_factor = 1.);
    void assign(const ScalarField & t_field, const double t_factor = 1.);
    void add(const VectorField & t_field, const double t_factor = 1.);
    void add(const ScalarField & t_field, const double t_factor = 1.);
    void subtract(const VectorField & t_field, const double t_factor = 1.);
    void subtract(const ScalarField & t_field, const double t_factor = 1.);
    void multiply(const VectorField & t_field, const double t_factor = 1.);
    void multiply(const ScalarField & t_field, const double t_factor = 1.);
    void divide(const VectorField & t_field, const double t_factor = 1.);
    void divide(const ScalarField & t_field, const double t_factor = 1.);
	void assignCrossProduct(const VectorField & t_field_1, const VectorField & t_field_2, const double t_factor = 1.);
	void addCrossProduct(const VectorField & t_field_1, const VectorField & t_field_2, const double t_factor = 1.);
	void assignProduct(const VectorField & t_field_1, const ScalarField & t_field_2, const double t_factor = 1.);
	void assignProduct(const ScalarField & t_field_1, const VectorField & t_field_2, const double t_factor = 1.);
	void addProduct(const VectorField & t_field_1, const ScalarField & t_field_2, const double t_factor = 1.);
	void addProduct(const ScalarField & t_field_1, const VectorField & t_field_2, const double t_factor = 1.);

	const double sumDotProduct(const VectorField & t_field);
	const double sumDotProduct(void);
    void enforceHermiticity(void);

	void conj(void);

    const VectorField & operator=(const VectorField & t_field);
    const VectorField & operator=(const ScalarField & t_field);
	const VectorField & operator=(const double t_x);
    const VectorField & operator+=(const VectorField & t_field);
    const VectorField & operator+=(const ScalarField & t_field);
    const VectorField & operator+=(double t_x);
    const VectorField & operator-=(const VectorField & t_field);
    const VectorField & operator-=(const ScalarField & t_field);
    const VectorField & operator-=(double t_x);
    const VectorField & operator*=(const VectorField & t_field);
    const VectorField & operator*=(const ScalarField & t_field);
    const VectorField & operator*=(double t_x);
    const VectorField & operator/=(const VectorField & t_field);
    const VectorField & operator/=(const ScalarField & t_field);
    const VectorField & operator/=(double t_x);

};

template<typename functional>
const VectorField operate(VectorField & t_field, const functional & f) {
    VectorField mesh;
    for (std::uint64_t i = 0; i < 3; i++)
        mesh[i] = operate(t_field[i], f);
    return mesh;
}

template<typename functional>
const VectorField operate(VectorField & t_field_1, VectorField & t_field_2, const functional & f) {
    VectorField mesh;
    for (std::uint64_t i = 0; i < 3; i++)
        mesh[i] = operate(t_field_1[i], t_field_2[i], f);
    return mesh;
}

#endif // *** VECTOR_FIELD_MESH_H *** //
