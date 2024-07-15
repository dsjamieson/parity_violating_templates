#include<parity_violating_template_generator.hpp>

int main(int argc, char ** argv) {
	int node_id;
	try {
        bool mpi_threads_ok;
        bool fftw_threads_ok;
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided); 
        MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
        if (argc != 3) {
            if (node_id == 0)
                std::cout << "Usage: parity_violating_template parameter_file num_threads\n";
            MPI_Finalize();
            exit(0);
        }
        const std::uint32_t num_threads = std::stoul(argv[2]);
        mpi_threads_ok = provided >= MPI_THREAD_FUNNELED;
        if (mpi_threads_ok) {
            omp_set_num_threads(num_threads);
            fftw_threads_ok = fftw_init_threads();
            if (fftw_threads_ok)
                fftw_plan_with_nthreads(num_threads);
            else if (node_id == 0)
                std::cout << "Warning, fftw_init_threads failed, continuing without OMP parallization for fftw\n";
        }
        else if (node_id == 0)
            std::cout << "Warning, MPI_Init_thread provided insufficient level of thread parallization, continuing without OMP parallization for fftw\n";
        fftw_mpi_init();
    }
	catch(...) {
		if (node_id == 0)
			std::cerr << "Error, failed to initialze MPI processes and/or OMP threads\n";
		MPI_Finalize();
		exit(-1);
	}
	try {
		std::string input_filename = argv[1];
        ParityViolatingTemplateGenerator generator(input_filename);
	}
	catch(std::runtime_error & t_error) {
		if (node_id == 0) {
			std::cerr << t_error.what();
			std::cerr << "Error, failed to generate data\n\n";
		}
		fftw_cleanup_threads();
		MPI_Finalize();
		exit(1);
	}
	fftw_cleanup_threads();
	MPI_Finalize();
    return 0;
}
