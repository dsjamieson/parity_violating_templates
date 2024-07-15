//************************
//*
//*
//*
//*
//************************

#ifndef INPUT_PARSER_H
#define INPUT_PARSER_H

#include<li_class.hpp>
#include<fstream>
#include<sstream>
#include<iterator>
#include<vector>
#include<algorithm>

class InputParser : public LIClass {

    public:

    InputParser(const std::string t_input_filename) {
		exception_prefix = "Error::InputParser, ";
        input_filename = t_input_filename;
    }

    void readInput(void) {
        std::ifstream input_file(input_filename);
		assert(input_file.is_open(), "could not open input parameter file for reading");
        std::string line;
        std::string::size_type split;
        while(getline(input_file, line)) {
            split = line.find_first_of("#%");
            if (split != std::string::npos)
                line.erase(split, line.size());
            if(line.empty())
                continue;
            std::istringstream entry_parser(line);
            std::vector<std::string> entry{std::istream_iterator<std::string>(entry_parser), std::istream_iterator<std::string>()};
            entries.push_back(entry);
        }
        input_file.close();
        return;
    }

    std::vector<std::vector<std::string>>::const_iterator findEntry(const std::string t_key) {
        return std::find_if(entries.begin(), entries.end(), [t_key](std::vector<std::string> t_e){return t_e[0] == t_key;});
    }

    template<typename t_type> t_type getParameter(const std::string t_key, const bool t_required, const t_type t_default) {
        std::vector<std::vector<std::string>>::const_iterator it = findEntry(t_key);
        assert(!(it == entries.end() && t_required), "no value given for required input parameter " + t_key);
		if (it != entries.end()) {
			if constexpr (std::is_same_v<double, t_type>)
				return stod((*it)[1]);
			else if constexpr (std::is_same_v<float, t_type>)
				return stof((*it)[1]);
			else if constexpr (std::is_same_v<std::uint64_t, t_type>)
				return stoull((*it)[1]);
			else if constexpr (std::is_same_v<std::int64_t, t_type>)
				return stoll((*it)[1]);
			else if constexpr (std::is_same_v<std::uint32_t, t_type>)
				return stoul((*it)[1]);
			else if constexpr (std::is_same_v<bool, t_type>)
				return stol((*it)[1]);
			else if constexpr (std::is_same_v<std::string, t_type>) {
				return (*it)[1];
			}
			else
				assert(false, "input parameter type not supported");
		}
        return t_default;
    }

    std::vector<std::vector<std::string>> entries;

    std::string input_filename;

};

#endif // *** INPUT_PARSER_H *** //
