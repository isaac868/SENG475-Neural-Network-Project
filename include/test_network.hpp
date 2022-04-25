#pragma once
#include "network.hpp"
#include "auxiliary_functions.hpp"
#include <istream>
#include <ostream>

void stream_test_data(int input_size, std::vector<VectorXd>& test_inputs, std::vector<std::string>& test_outputs, std::istream& stream);

void output_results(const std::vector<VectorXd>& test_inputs, const std::vector<std::string>& test_outputs, const ml::neural_network& nn, std::ostream& stream);