#include<fields/vector_field.hpp>
#include<utils/input_parser.hpp>
#include<utils/random.hpp>
#include<utils/io_hdf5.hpp>
#include<utils/timer.hpp>
#include<filesystem>

class ParityViolatingTemplateGenerator : public LIClass {

    public :

    ParityViolatingTemplateGenerator(void) {
		exception_prefix = "Error::ParityViolatingTemplateGenerator, ";
		label_string = "partity violating template generator";
	}

    ParityViolatingTemplateGenerator(std::string t_input_filename) : ParityViolatingTemplateGenerator() {
		InputParser parser(t_input_filename);
		parser.readInput();
		output_directory = parser.getParameter<std::string>("OutputDirectory", false, "./");
		if (output_directory.back() != '/')
			output_directory += "/";
        std::filesystem::create_directories(output_directory);
		output_filename_base = parser.getParameter<std::string>("OutputFilenameBase", true, "");
        random_seed_filename = parser.getParameter<std::string>("RandomSeedFilename", false, "");
	    num_mocks = parser.getParameter<std::uint64_t>("NumMocks", false, 1);
	    start_mock = parser.getParameter<std::uint64_t>("StartMock", false, 0);
		num_mesh_1d = parser.getParameter<std::uint64_t>("NumMesh1D", true, 0);
		box_length = parser.getParameter<double>("BoxLength", true, 1.);
		box_volume = box_length * box_length * box_length;
		sphere_cutoff = parser.getParameter<bool>("SphereCutoff", false, false);
        wave_number_cutoff = parser.getParameter<double>("WaveNumberCutoff", false, 0.);
		fixed_amplitude = parser.getParameter<bool>("FixedAmplitude", false, false);
		phase_flip = parser.getParameter<bool>("PhaseFlip", false, false);

        scalar_amplitude = parser.getParameter<double>("ScalarAmplitude", true, 1.);
        spectral_tilt = parser.getParameter<double>("SpectralTilt", true, 1.);
        pivot_wave_number = parser.getParameter<double>("PivotWaveNumber", false, 0.05);
        template_exponent_a = parser.getParameter<double>("TemplateExponentA", true, 1.);
        template_exponent_b = parser.getParameter<double>("TemplateExponentB", true, 1.);
        template_exponent_c = parser.getParameter<double>("TemplateExponentC", true, 1.);


        if (num_mocks > 1)
            timer.setMaxMessageLength("Generating mock " + std::to_string(start_mock + num_mocks) + " of " + std::to_string(num_mocks)  + "... ");
        else
            timer.setMaxMessageLength("Generating mock... ");


        timer.printStart("Initializing...");

        linear_potential.initialize(num_mesh_1d, box_length);
        nonlinear_potential.initialize(num_mesh_1d, box_length);
        vector_field_a.initialize(num_mesh_1d, box_length);
        vector_field_b.initialize(num_mesh_1d, box_length);
        vector_field_c.initialize(num_mesh_1d, box_length);

        wave_number_cutoff =  (wave_number_cutoff == 0.) ? linear_potential.nyquist_wave_number : wave_number_cutoff;

        random_generator = std::make_unique<Random>(simulation_seeds);
        seed_table.resize(num_mesh_1d * num_mesh_1d, std::vector<std::uint64_t>(std::mt19937::state_size));

        timer.printDone();

        for (std::uint64_t mock = start_mock; mock < start_mock + num_mocks; mock++) {
            if (num_mocks > 1)
                timer.printStart("Generating mock " + std::to_string(mock) + " of " + std::to_string(num_mocks) + "...");
            else
                timer.printStart("Generating mock...");
            getRandomSeeds(mock, mock == start_mock);
            constructTemplate(template_exponent_a, template_exponent_b, template_exponent_c, scalar_amplitude, spectral_tilt);
            saveTemplate(mock);
            timer.printDone();
        }

    }

    Timer timer;

    std::string random_seed_filename;
    std::string output_directory;
    std::string output_filename_base;
    std::uint64_t num_mocks;
    std::uint64_t start_mock;

    std::uint64_t num_mesh_1d;
    double box_length;
    double box_volume;
    bool sphere_cutoff;
    double wave_number_cutoff;
    bool fixed_amplitude;
    bool phase_flip;

    ScalarField linear_potential;
    ScalarField nonlinear_potential;
    VectorField vector_field_a;
    VectorField vector_field_b;
    VectorField vector_field_c;
    double scalar_amplitude;
    double spectral_tilt;
    double pivot_wave_number;
    double template_exponent_a; 
    double template_exponent_b; 
    double template_exponent_c;

    std::unique_ptr<Random> random_generator;
    std::vector<std::uint64_t> simulation_seeds;
    std::vector<std::vector<std::uint64_t>> seed_table;

    void initializeRandomGaussianModes(const double t_scalar_amplitude, const double t_spectral_tilt) {
        ScalarField * f = &linear_potential;
        #pragma omp parallel
        {
            const double pi = acos(-1.);
            const double two_pi = 2. * pi;
            const double amplitude_norm = sqrt(scalar_amplitude * two_pi * pi * f->box_volume);
            const double tilt = 0.5 * (spectral_tilt - 1.);
            std::uint64_t seed_table_index;
            std::uint64_t mode_index;
            Random thread_rng;
            double k_mag;
            double amplitude;
            double phase;
            #pragma omp for collapse(2)
            for (std::uint64_t i = f->local_start_x; i < f->local_end_x; i++) {
                for (std::uint64_t j = 0; j < f->num_mesh_1d; j++) {
                    seed_table_index = f->getFieldIndex(0, i, j);
                    thread_rng.seedRandomGenerator(seed_table[seed_table_index]);
                    for (std::uint64_t k = 0; k < f->num_modes_last_d - 1; k++) {
                        k_mag = sqrt(f->wave_numbers[i] * f->wave_numbers[i] + f->wave_numbers[j] * f->wave_numbers[j] + f->wave_numbers[k] * f->wave_numbers[k]);
                        if (k_mag == 0)
                                continue;
                        mode_index = f->getLocalModeIndex(i, j, k);
                        if (sphere_cutoff && k_mag >= wave_number_cutoff) {
                            linear_potential.modes[mode_index] =  0.;
                            continue;
                        }
                        else if (f->wave_numbers[i] > wave_number_cutoff || f->wave_numbers[j] > wave_number_cutoff || f->wave_numbers[k] > wave_number_cutoff) {
                            linear_potential.modes[mode_index] =  0.;
                            continue;
                        }
                        amplitude = 0.;
                        while (amplitude == 0.)
                            amplitude = thread_rng.getRandomUniform();
                        amplitude = sqrt(-log(amplitude));
                        amplitude = fixed_amplitude ? 1. : amplitude;
                        amplitude *= pow(k_mag, -1.5) * amplitude_norm;
                        amplitude = (spectral_tilt == 1.) ? amplitude : amplitude * pow(k_mag / pivot_wave_number, tilt);
                        phase = two_pi * thread_rng.getRandomUniform();
                        phase = phase_flip ? phase - two_pi : phase; 
                        linear_potential.modes[mode_index] = amplitude * std::complex(cos(phase), sin(phase));
                    }
                }
            }
        }
        linear_potential.enforceHermiticity();
        return;
    }

    void filterModes(ScalarField & t_modes, const double t_filter_exponent, ScalarField & t_filtered_modes) {
        ScalarField * f = &t_modes;
        #pragma omp parallel
        {
            std::uint64_t mode_index;
            double k_mag;
            #pragma omp for collapse(3)
            for (std::uint64_t i = f->local_start_x; i < f->local_end_x; i++) {
                for (std::uint64_t j = 0; j < f->num_mesh_1d; j++) {
                    for (std::uint64_t k = 0; k < f->num_modes_last_d; k++) {
                        k_mag = sqrt(f->wave_numbers[i] * f->wave_numbers[i] + f->wave_numbers[j] * f->wave_numbers[j] + f->wave_numbers[k] * f->wave_numbers[k]);
                        mode_index = f->getLocalModeIndex(i, j, k);
                        if (k_mag == 0) {
                            t_filtered_modes.modes[mode_index] = 0.;
                            continue;
                        }
                        t_filtered_modes.modes[mode_index] = t_modes.modes[mode_index] * pow(k_mag, t_filter_exponent);
                    }
                }
            }
        }
        return;
    }

    void constructTemplate(const double t_a, const double t_b, const double t_c, const double t_scalar_amplitude, const double t_spectral_tilt) {
        template_exponent_a = t_a;
        template_exponent_b = t_b;
        template_exponent_c = t_c;
        scalar_amplitude = t_scalar_amplitude;
        spectral_tilt = t_spectral_tilt;
        initializeRandomGaussianModes(t_scalar_amplitude, t_spectral_tilt);
        filterModes(linear_potential, template_exponent_a - 1, nonlinear_potential);
        nonlinear_potential.getGradientModes(vector_field_a);
        vector_field_a.transformInverseDFT();
        filterModes(linear_potential, template_exponent_b - 1, nonlinear_potential);
        nonlinear_potential.getGradientModes(vector_field_b);
        vector_field_b.transformInverseDFT();
        vector_field_c.assignCrossProduct(vector_field_a, vector_field_b);
        filterModes(linear_potential, template_exponent_c - 1, nonlinear_potential);
        nonlinear_potential.getGradientModes(vector_field_b);
        vector_field_b.transformInverseDFT();
        nonlinear_potential.assignDotProduct(vector_field_c, vector_field_b);
        linear_potential.transformInverseDFT();
        return;
    }

    void writeHeader(const std::string t_filename) {
        H5OutFile file(t_filename);
        const std::uint64_t num_files = getNumNodes();
        file.writeFileAttribute("NumFiles", num_files);
        file.writeFileAttribute("NumMesh1D", num_mesh_1d);
        file.writeFileAttribute("BoxLength", box_length);
        file.writeFileAttribute("ScalarAmplitude", scalar_amplitude);
        file.writeFileAttribute("SpectralTilt", spectral_tilt);
        file.writeFileAttribute("TemplateExponentA", template_exponent_a);
        file.writeFileAttribute("TemplateExponentB", template_exponent_b);
        file.writeFileAttribute("TemplateExponentC", template_exponent_c);
        file.writeToDataset("RandomSeed", simulation_seeds);
        return;
    }

    void writeScalarField(const std::string t_filename, const std::string t_datasetname, const ScalarField & t_field) {
        H5AppFile file(t_filename);
        if(!file.openDataset(t_datasetname)) {
            hsize_t ndims = 3;
            std::vector<hsize_t> dims(ndims, t_field.num_mesh_1d);
            dims[0] = t_field.local_num_x;
            file.createDataspace(dims);
            dims[0] = 1;
            dims[1] = 1;
            file.createPropertyList(dims);
            file.createDataset<float>(t_datasetname);
        }
        hsize_t ndims = 3;
        std::vector<hsize_t> start(ndims, 0);
        std::vector<hsize_t> count(ndims, 1);
        std::uint64_t col_index;
        std::vector<float> buffer(t_field.num_mesh_1d);
        std::uint64_t i_local;
        for (std::uint64_t i = t_field.local_start_x; i < t_field.local_end_x; i++) {
            i_local = i - t_field.local_start_x;
            for (std::uint64_t j = 0; j < t_field.num_mesh_1d; j++) {
                col_index = t_field.getLocalMeshIndex(i, j, 0);
                start = std::vector<hsize_t>({i_local, j, 0});
                count = std::vector<hsize_t>({1, 1, t_field.num_mesh_1d});
                std::copy(t_field.begin() + col_index, t_field.begin() + col_index + t_field.num_mesh_1d, buffer.begin());
                file.selectHyperslab(start, count);
                file.writeToDataset(buffer);
            }
        }
        return;
    }
    
    std::string getFilename(std::uint64_t t_mock) {
        return output_directory + output_filename_base + "_" + std::to_string(t_mock) + ".hdf5";
    }

    void saveTemplate(const std::uint64_t t_mock) {
        std::string filename = getFilename(t_mock);
        writeHeader(filename);
        writeScalarField(filename, "LinearPotential", linear_potential);
        writeScalarField(filename, "NonlinearPotential", nonlinear_potential);
        return;
    }

    void getRandomSeeds(const std::uint64_t t_mock, const bool t_first) {
		if (random_seed_filename.size() != 0) {
			try {
				H5InFile in_seed_file(random_seed_filename);
                std::string dataset_name = "RandomSeed" + std::to_string(t_mock);
				in_seed_file.readDataset(dataset_name, simulation_seeds);
				random_generator->seedRandomGenerator(simulation_seeds);
			}
			catch(std::runtime_error & t_error) {
				std::cerr << t_error.what();
				assert(false, "failed to read random seeds from InputRandomSeedsFilename " + random_seed_filename);
			}
		}
        else {
            random_generator->generateRandomSeeds(simulation_seeds);
            random_generator->seedRandomGenerator(simulation_seeds);
            writeRandomSeeds(t_mock, t_first);
            for (auto & seeds : seed_table)
                random_generator->drawRandomSeeds(seeds);
        }
        return;
    }

    void writeRandomSeeds(const std::uint64_t t_mock, const bool t_first) {
        std::string filename = output_directory + "random_seeds.hdf5";
        if (t_first) {
            try {
                H5OutFile out_seed_file(filename);
                std::string dataset_name = "RandomSeed" + std::to_string(t_mock);
                out_seed_file.writeToDataset(dataset_name, simulation_seeds);
            }
            catch(std::runtime_error & t_error) {
                std::cerr << t_error.what();
            }
        }
        else {
            try {
                H5AppFile out_seed_file(filename);
                std::string dataset_name = "RandomSeed" + std::to_string(t_mock);
                out_seed_file.writeToDataset<std::uint64_t>(dataset_name, simulation_seeds);
            }
            catch(std::runtime_error & t_error) {
                std::cerr << t_error.what();
            }
        }
        return;
    }

};
