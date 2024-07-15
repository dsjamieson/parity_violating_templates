//************************
//*
//*
//*
//*
//************************

#ifndef IO_HDF5_H
#define IO_HDF5_H

#include<li_class.hpp>
#include<hdf5.h>
#include<mpi.h>

class H5File : public LIClass {

	public :

    H5File(const bool t_suppress_errors = true) {
        if (t_suppress_errors)
            H5Eset_auto(H5E_DEFAULT, NULL, NULL);
		exception_prefix = "Error::H5File, ";
    }

    ~H5File(void) {
		closeAttribute();
		closePropertyList();
		closeDataspace();
		closeMemoryspace();
		closeDataset();
		closeGroup();
		closeFile();
    }

	MPI_Comm mpi_comm;;
    MPI_Info mpi_info;

    hid_t file = 0;
    hid_t group = 0;  
    hid_t dataset = 0;
	hid_t dataspace = 0;
	hid_t memoryspace = 0;
	hid_t property_list = 0;
    hid_t attribute = 0;
	hid_t datatype = 0;

	hsize_t ndims;
	std::vector<hsize_t> dims;

	void closeFile(void) {
        if (file > 0)
			assert(H5Fclose(file) >= 0, "file did not close successfully");
		file = 0;
		return;
	}

	void closeGroup(void) {
        if (group > 0)
			assert(H5Gclose(group) >= 0, "group did not close successfully");
		group = 0;
		return;
	}

	void closeDataset(void) {
        if (dataset > 0)
			assert(H5Dclose(dataset) >= 0, "dataset did not close successfully");
		dataset = 0;
		return;
	}

	void closeDataspace(void) {
        if (dataspace > 0)
			assert(H5Sclose(dataspace) >= 0, "dataspace did not close successfully");
		dataspace = 0;
		return;
	}

	void closeMemoryspace(void) {
        if (memoryspace > 0)
			assert(H5Sclose(memoryspace) >= 0, "memoryspace did not close successfully");
		memoryspace = 0;
		return;
	}

	void closePropertyList(void) {
        if (property_list > 0)
			assert(H5Pclose(property_list) >= 0, "property_list did not close successfully");
		property_list = 0;
		return;
	}

	void closeAttribute(void) {
        if (attribute > 0)
			assert(H5Aclose(attribute) >= 0, "attribute did not close successfully");
		attribute = 0;
		return;
	}

	void assertValidID(const hid_t t_id, const std::string t_error = "") {
		assert(t_id != H5I_INVALID_HID, t_error);
		return;
	}

	void assertSuccess(const hid_t t_code, const std::string t_error = "") {
		assert(t_code >= 0, t_error);
		return;
	}

	template<typename t_type>
	const hid_t getDataType(void) {
		if (std::is_same<double, t_type>::value)
			return H5T_NATIVE_DOUBLE;
		else if (std::is_same<float, t_type>::value)
			return H5T_NATIVE_FLOAT;
		else if (std::is_same<std::uint64_t, t_type>::value)
			return H5T_STD_U64LE;
		else if (std::is_same<std::uint32_t, t_type>::value)
			return H5T_STD_U32LE;
		else if (std::is_same<std::int64_t, t_type>::value)
			return H5T_STD_I64LE;
		else if (std::is_same<std::int32_t, t_type>::value)
			return H5T_STD_I32LE;
		else {
			assert(false, "invalid data type encountered");
		}
		return H5I_INVALID_HID;
	}

};

class H5InFile : public H5File {

    public :

	H5InFile(const bool t_suppress_errors = true) : H5File(t_suppress_errors) {
		exception_prefix = "Error::H5InFile, ";
	}

    H5InFile(const std::string t_filename, const bool t_suppress_errors = true) : H5InFile(t_suppress_errors) {
        openFile(t_filename);
    }

    void openFile(const std::string t_filename) {
		closeFile();
        assertValidID(file = H5Fopen(t_filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), "file " + t_filename + " was not opened");
        return;
    }

    void openGroup(const std::string t_groupname) {
		closeGroup();
        assertValidID(group = H5Gopen(file, t_groupname.c_str(), H5P_DEFAULT), "group " + t_groupname + " was not opened");
        return;
    }

    void openDataset(const std::string t_datasetname) {
        closeDataset();
        dataset = H5Dopen(file, t_datasetname.c_str(), H5P_DEFAULT);
		assert(dataset != H5I_INVALID_HID, "dataset " + t_datasetname + " was not opened");
		openDataspace();
        return;
    }

    void openGroupDataset(const std::string t_datasetname) {
		assert(group != 0, "a group must be open to open a group dataset");
        closeDataset();
        dataset = H5Dopen(group, t_datasetname.c_str(), H5P_DEFAULT);
		assert(dataset != H5I_INVALID_HID, "group dataset " + t_datasetname + " was not opened");
		openDataspace();
        return;
    }

	void openDataspace(void) {
		closeDataspace();
		dataspace = H5Dget_space(dataset);
		assert(dataspace != H5I_INVALID_HID, "dataspace was not opened");
		ndims = H5Sget_simple_extent_ndims(dataspace);
		dims.resize(ndims);
		H5Sget_simple_extent_dims(dataspace, dims.data(), NULL);
		return;
	}

    void openGroupAttribute(const std::string t_attributename) {
		assert(group != 0, "a group must be open to open a group attribute");
        closeAttribute();
        attribute = H5Aopen(group, t_attributename.c_str(), H5P_DEFAULT);
		assert(attribute != H5I_INVALID_HID, "group attribute " + t_attributename + " was not opened");
        return;
    }

    void openDatasetAttribute(const std::string t_attributename) {
		assert(dataset != 0, "a dataset must be open to open a dataset attribute");
        closeAttribute();
        attribute = H5Aopen(dataset, t_attributename.c_str(), H5P_DEFAULT);
		assert(attribute != H5I_INVALID_HID, "dataset attribute " + t_attributename + " was not opened");
        return;
    }

	template<typename t_type>
	void readDataset(std::vector<t_type> & t_vector, const bool t_resize = false) {
		assert(dataset > 0, "a dataset must be opened before reading");
		datatype = getDataType<t_type>();
		assert(H5Tequal(datatype, H5Dget_type(dataset)) > 0, "inconsistent vector and dataset datatype");
		hsize_t ndims = H5Sget_simple_extent_ndims(dataspace);
		hsize_t dims[ndims], maxdims[ndims];
		H5Sget_simple_extent_dims(dataspace, dims, maxdims);
		std::uint64_t total_size = 1;
		for (std::uint64_t i = 0; i < ndims; i++) 
			total_size *= dims[i];
		if (t_resize)
			t_vector.resize(total_size);
		else
			assert(total_size == t_vector.size(), "inconsistent vector and dataset sizes\n");
		assert(H5Dread(dataset, datatype,  H5S_ALL, H5S_ALL, H5P_DEFAULT, t_vector.data()) >= 0, "failed to read H5 Dataset");
		return;
	}

	template<typename t_type>
	void readDataset(const std::string t_datasetname, std::vector<t_type> & t_vector, const bool t_resize = false) {
		openDataset(t_datasetname);
		readDataset(t_vector, t_resize);
		return;
	}

	template<typename t_type>
	void readDataset(const std::string t_filename, const std::string t_datasetname, std::vector<t_type> & t_vector, const bool t_resize = false) {
		openFile(t_datasetname);
		readDataset(t_datasetname, t_vector, t_resize);
		return;
	}

	template<typename t_type>
	void readGroupDataset(const std::string t_groupname, const std::string t_datasetname, std::vector<t_type> & t_vector) {
		openGroup(t_groupname);
		openGroupDataset(t_datasetname);
		readDataset(t_vector);
		return;
	}

	template<typename t_type>
	void readGroupDataset(const std::string t_filename, const std::string t_groupname, const std::string t_datasetname, std::vector<t_type> & t_vector) {
		openFile(t_filename);
		readGroupDataset(t_groupname, t_datasetname, t_vector);
		return;
	}

	void selectHyperslab(const std::vector<hsize_t> t_start, const std::vector<hsize_t> t_count) {
		dataspace = H5Dget_space(dataset);
		H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, t_start.data(), NULL, t_count.data(), NULL);
		createMemoryspace(t_count);
		return;
	}

	void createMemoryspace(const std::vector<hsize_t> t_dims) {
		closeMemoryspace();
		memoryspace = H5Screate_simple(t_dims.size(), t_dims.data(), NULL);
        assert(memoryspace != H5I_INVALID_HID, "failed to create memoryspace");
		return;
	}

	template<typename t_type>
	void readDatasetHyperslab(const std::vector<hsize_t> t_start, const std::vector<hsize_t> t_count, std::vector<t_type> & t_vector) {
		selectHyperslab(t_start, t_count);
		readDatasetHyperslab(t_vector);
		return;
	}

	template<typename t_type>
	void readDatasetHyperslab(const std::vector<t_type> & t_vector, const bool t_resize = false) {
		assert(dataset > 0, "a dataset must be opened before reading hyperslab");
		assert(dataspace > 0, "a dataspace must be opened before reading hyperslab");
		assert(memoryspace > 0, "a memoryspace must be opened before reading hyperslab");
		datatype = getDataType<t_type>();
		assert(H5Tequal(datatype, H5Dget_type(dataset)) > 0, "inconsistent vector and dataset hyperslab datatype");
		hsize_t ndims = H5Sget_simple_extent_ndims(dataspace);
		assert(ndims == 1, "dataset hyperslab has more than one dimension, cannot read into vector");
		hsize_t dims[ndims], maxdims[ndims];
		H5Sget_simple_extent_dims(dataspace, dims, maxdims);
		if (t_resize)
			t_vector.resize(dims[0]);
		else
			assert(dims[0] == t_vector.size(), "inconsistent vector and dataset sizes");
		assert(H5Dread(dataset, datatype, dataspace, memoryspace, H5P_DEFAULT, t_vector.data()) >= 0, "failed to read H5 Dataset");
		return;
	}

	template<typename t_type>
	void readFileAttribute(const std::string t_attributename, t_type & t_value) {
		assert(file != 0, "a file must be open before reading a file attribute");
        assertValidID(attribute = H5Aopen(file, t_attributename.c_str(), H5P_DEFAULT), "failed to open H5 file attribute");
		datatype = getDataType<t_type>();
		assert(H5Tequal(datatype, H5Aget_type(attribute)) > 0, "inconsistent value and attribute datatype");
		assert(H5Aread(attribute, datatype, &t_value) >= 0, "failed to read H5 attribute");
		return;
	}

};

class H5OutFile : public H5InFile {

    public :

    H5OutFile(const bool t_suppress_errors = true) : H5InFile(t_suppress_errors) {
		exception_prefix = "Error::H5OutFile, ";
    }

    H5OutFile(const std::string t_filename, const bool t_suppress_errors = true) : H5OutFile(t_suppress_errors) {
        createFile(t_filename);
    }

    void createFile(const std::string t_filename) {
        closeFile();
		file = H5Fcreate(t_filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        assert(file != H5I_INVALID_HID, "failed to create file " + t_filename);
        return;
    }

    void createGroup(const std::string t_groupname) {
        closeGroup();
		group = H5Gcreate2(file, t_groupname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        assert(group != H5I_INVALID_HID, "Error, failed to create group " + t_groupname);
        return;
    }

	void createDataspace(const std::vector<hsize_t> t_dims, const std::vector<hsize_t> t_max_dims = {}) {
		closeDataspace();
		ndims = t_dims.size();
		dims = t_dims;
		std::vector<hsize_t> max_dims(ndims);
		if (t_max_dims.size() == 0)
			std::fill(max_dims.begin(), max_dims.end(), H5S_UNLIMITED);
		else {
			assert(t_max_dims.size() == ndims, "incompatible number of entries in dims and max_dims for creating dataspace");
			std::copy(t_max_dims.begin(), t_max_dims.end(), max_dims.begin());
		}
		dataspace = H5Screate_simple(ndims, dims.data(), max_dims.data());
        assert(dataspace != H5I_INVALID_HID, "failed to create dataspace");
		return;
	}

	void createPropertyList(const std::vector<hsize_t> t_chunk_dims) {
		closePropertyList();
		property_list = H5Pcreate(H5P_DATASET_CREATE);
		assert(property_list != H5I_INVALID_HID, "failed to create property list");
		assert(H5Pset_layout(property_list, H5D_CHUNKED) >= 0, "failed to set property_list layout");
		assert(H5Pset_chunk(property_list, t_chunk_dims.size(), t_chunk_dims.data()) >= 0, "failed to set property list chunks");
		return;
	}

	template<typename t_type>
	void createDataset(const std::string t_datasetname) {
		assert(file > 0, "a file must be open before creating a dataset");
		assert(dataspace > 0, "a dataspace must be open before creating an dataset");
		assert(property_list > 0, "a property list must be created before creating an dataset");
		closeDataset();
		datatype = getDataType<t_type>();
		dataset = H5Dcreate2(file, t_datasetname.c_str(), datatype, dataspace, H5P_DEFAULT, property_list, H5P_DEFAULT);
		closePropertyList();
		assert(dataset != H5I_INVALID_HID, "failed to create dataspace");
		return;
	}

	template<typename t_type>
    void createGroupDataset(const std::string t_datasetname) {
		assert(group > 0, "a group must be created before creating a group dataset");
		assert(dataspace > 0, "a dataspace must be created before creating an dataset");
		closeDataset();
		datatype = getDataType<t_type>();
        dataset = H5Dcreate2(group, t_datasetname.c_str(), datatype, dataspace, H5P_DEFAULT, property_list, H5P_DEFAULT);
        assert(dataset != H5I_INVALID_HID, "Error, failed to create group dataset " + t_datasetname);
        return;
    }

	template<typename t_type>
    void createGroupDataset(const std::string t_group, std::string t_datasetname) {
		createGroup(t_group);
		createGroupDataset<t_type>(t_datasetname);
        return;
    }

	void setDatasetExtent(const std::vector<hsize_t> t_dims) {
		H5Dset_extent(dataset, t_dims.data());
		return;
	}

	template<typename t_type>
	void writeToDataset(const std::string t_datasetname,  const std::vector<t_type> & t_vector, const std::uint64_t t_chunk_size = 1) {
		createDataspace({t_vector.size()});
		const hsize_t chunk_size = (t_chunk_size == 0) ? t_vector.size() : t_chunk_size;
		createPropertyList({chunk_size});
		createDataset<t_type>(t_datasetname);
		createMemoryspace({t_chunk_size});
		selectHyperslab({0}, {t_vector.size()});
        assert(H5Dwrite(dataset, datatype, memoryspace, dataspace, H5P_DEFAULT, t_vector.data()) >= 0, "failed to write dataset");
		return;
	}

	template<typename t_type>
	void writeToDataset(const std::string t_datasetname, const t_type & t_value, const std::uint64_t t_chunk_size = 1) {
		createDataspace({1});
		createPropertyList({t_chunk_size});
		createDataset<t_type>(t_datasetname);
		createMemoryspace({t_chunk_size});
		selectHyperslab({0}, {1});
        assert(H5Dwrite(dataset, datatype, memoryspace, dataspace, H5P_DEFAULT, &t_value) >= 0, "failed to write dataset");
		return;
	}

	template<typename t_type>
	void writeVectorToGroupDataset(const std::string t_datasetname, const std::vector<t_type> & t_vector) {
		createDataspace({t_vector.size()});
		createGroupDataset<t_type>(t_datasetname);
        assert(H5Dwrite(dataset, datatype, memoryspace, dataspace, H5P_DEFAULT, t_vector.data()) >= 0, "failed to write group dataset");
		return;
	}

	template<typename t_type>
	void writeToGroupDataset(const std::string t_datasetname, const std::vector<t_type> & t_value) {
		createDataspace({1});
		createGroupDataset<t_type>(t_datasetname);
        assert(H5Dwrite(dataset, datatype, memoryspace, dataspace, H5P_DEFAULT, &t_value) >= 0, "failed to write group dataset");
		return;
	}

	template<typename t_type>
	void writeToGroupDataset(const std::string t_groupname, const std::string t_datasetname, const t_type & t_data) {
		createGroup(t_groupname);
		writeVectorToGroupDataset(t_datasetname, t_data);
		return;
	}
 
	template<typename t_type>
	void createFileAttribute(const std::string t_attributename) {
		assert(file != 0, "a file must be open before writing a file attribute");
		assertValidID(property_list = H5Pcreate(H5P_ATTRIBUTE_CREATE), "failed to create file attribute property list");
		assertValidID(dataspace = H5Screate(H5S_SCALAR), "failed to create file attribute dataspace"); 
		datatype = getDataType<t_type>();
        assertValidID(attribute = H5Acreate2(file, t_attributename.c_str(), datatype, dataspace, property_list, H5P_DEFAULT),
			"failed to create file attribute");
		return;
	}

	template<typename t_type>
	void writeFileAttribute(const std::string t_attributename, const t_type & t_value) {
		createFileAttribute<t_type>(t_attributename);
		datatype = getDataType<t_type>();
		assertSuccess(H5Awrite(attribute, datatype, &t_value), "failed to write file attribute " + t_attributename);
		return;
	}

};

class H5AppFile : public H5OutFile {

    public :

    using H5OutFile::writeToDataset;

    H5AppFile(const bool t_suppress_errors = true) : H5OutFile(t_suppress_errors) {
		exception_prefix = "Error::H5AppFile, ";
    }

    H5AppFile(const std::string t_filename, bool t_suppress_errors = true) : H5AppFile(t_suppress_errors) {
        openFile(t_filename);
    }

    void openFile(const std::string t_filename) {
        closeFile();
		file = H5Fopen(t_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		if (file == H5I_INVALID_HID)
			createFile(t_filename);
        return;
    }

    void openGroup(const std::string t_groupname) {
		closeGroup();
        group = H5Gopen(file, t_groupname.c_str(), H5P_DEFAULT);
		if (group == H5I_INVALID_HID)
			createGroup(t_groupname);
        return;
    }

    bool openDataset(const std::string t_datasetname) {
        closeDataset();
        dataset = H5Dopen(file, t_datasetname.c_str(), H5P_DEFAULT);
		if (dataset == H5I_INVALID_HID)
			return false;
		openDataspace();
        return true;
    }

    bool openGroupDataset(const std::string t_datasetname) {
        closeDataset();
        dataset = H5Dopen(group, t_datasetname.c_str(), H5P_DEFAULT);
		if (dataset == H5I_INVALID_HID)
			return false;
		openDataspace();
        return true;
    }

	template<typename t_type>
	void writeToDataset(const std::vector<t_type> t_vector) {
		assert(dataset > 0, "a dataset must be open before writing data");
		assert(memoryspace > 0, "a memory space must be open before writing");
		assert(dataspace > 0, "a dataspace must be open before writing");
		datatype = getDataType<t_type>();
		assert(H5Tequal(datatype, H5Dget_type(dataset)) > 0, "inconsistent value and dataset types for appending single value");
        assert(H5Dwrite(dataset, datatype, memoryspace, dataspace, H5P_DEFAULT, t_vector.data()) >= 0, "failed to write dataset");
		return;
	}

	template<typename t_type>
	void writeToDataset(const std::string t_datasetname, const t_type t_value) {
		assert(openDataset(t_datasetname), "failed to open dataset " + t_datasetname + " for appending");
		datatype = getDataType<t_type>();
		assert(H5Tequal(datatype, H5Dget_type(dataset)) > 0, "inconsistent value and dataset types for appending single value");
		assert(ndims == 1, "inconsistent dataset ndims for appending single value");
		std::vector<hsize_t> start = dims;
		std::vector<hsize_t> count(ndims, 1);
		std::vector<hsize_t> new_dims = dims;
		new_dims[0]++;
		setDatasetExtent(new_dims);
		selectHyperslab(start, count);
        assert(H5Dwrite(dataset, datatype, memoryspace, dataspace, H5P_DEFAULT, &t_value) >= 0, "failed to write dataset");
		return;
	}

	template<typename t_type>
	void writeToDataset(const std::string t_datasetname, const t_type t_value, const hsize_t t_chunk_size) {
		if (!openDataset(t_datasetname)) {
			createDataspace({0});
			createPropertyList({t_chunk_size});
			createDataset<t_type>(t_datasetname);
		}
		writeToDataset(t_datasetname, t_value);
		return;
	}

};


#endif // *** IO_HDF5_H *** //
