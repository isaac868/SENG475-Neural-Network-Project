#pragma once
#include <vector>
#include <istream>
#include <string>
#include <algorithm>
#include <Eigen/Dense>
#include <set>

using namespace Eigen;

std::vector<std::string> input_csv(std::istream& stream);

MatrixXd readMatrix(std::istream& stream, int rows, int cols);

void stream_network_data(std::vector<MatrixXd>& weights, std::vector<VectorXd>& biases, std::set<std::string>& labels, std::istream& stream);