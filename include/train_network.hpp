#pragma once
#include "network.hpp"
#include "auxiliary_functions.hpp"
#include <istream>
#include <ostream>

namespace ml
{

std::istream& operator>>(std::istream& stream, ml::topology& topo);

}

void stream_training_data(std::vector<std::pair<VectorXd, std::string>>& training_data, std::istream& stream);

// Ensures number of unique labels in training_data matches what is expected by the topology and that the input dimension of
// training_data matches the topology
void validate_topology_requirements(int first_layer_size, int last_layer_size, const std::vector<std::pair<VectorXd, std::string>>& training_data);