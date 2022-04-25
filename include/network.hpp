#pragma once
#include <vector>
#include <set>
#include <Eigen/Dense>
#include <utility>
#include <ostream>
#include <map>
#include <atomic>
#include <mutex>
#include <span>

namespace ml
{

using namespace Eigen;

struct topology
{
    std::vector<int> layers;
};

std::ostream& operator<<(std::ostream& stream, const topology& topo);

class neural_network
{
private:
    std::vector<MatrixXd>                 m_weights;
    std::vector<VectorXd>                 m_biases;
    topology                              m_topo;
    const std::map<std::string, VectorXd> m_labels;
    std::mutex                            mtx;
    // Ideally m_latest_weight_gradients and m_latest_bias_gradients would not be properties of the class,
    // however back_propagate's function signature was getting too large
    std::vector<MatrixXd>                 m_latest_weight_gradients;
    std::vector<VectorXd>                 m_latest_bias_gradients;

    static void hadamard_product(VectorXd& lhs, const VectorXd& rhs);

    static void logistic(VectorXd& vector);
    static void logistic_derivative(VectorXd& vector);

    [[nodiscard]] static std::map<std::string, VectorXd> generate_class_vector_map(std::set<std::string>);

public:
    neural_network(topology topo, std::set<std::string> labels);
    neural_network(std::vector<MatrixXd> weights, std::vector<VectorXd> biases, std::set<std::string> labels);

    void randomize();

    // returns the largest activation of the final layer and it's coresponding class
    [[nodiscard]] std::pair<std::string, double>                classify(const VectorXd& input_activations) const;
    // Generates a confusion matrix with the columns representing predicted labels and the rows representing actual labels, vector component
    // of pair is to indicate ordering of labels in rows and columns
    [[nodiscard]] std::pair<std::vector<std::string>, MatrixXi> generate_confusion_matrix(const std::vector<std::pair<VectorXd, std::string>>& test_data) const;
    // returns a vector of all calculated activations and weighted sums
    [[nodiscard]] std::vector<std::pair<VectorXd, VectorXd>>    forward_propagate_return_all(const VectorXd& input_activations, const std::vector<MatrixXd>& local_weights, const std::vector<VectorXd>& local_biases) const;

    // performes back propagation. training_data is a vector of pairs of training input data and desired output classification
    double back_propagate(const std::vector<std::pair<VectorXd, VectorXd>>& training_data, const std::span<int>& mini_batch, double learning_rate);
    // trains network
    void   train(const std::vector<std::pair<VectorXd, std::string>>& training_data, bool print_cost, std::atomic_bool& stopped, double learning_rate = 0.1, std::size_t mini_batch_size = 0);

    friend std::ostream& operator<<(std::ostream&, const neural_network&);
};

std::ostream& operator<<(std::ostream& stream, const neural_network& network);

};