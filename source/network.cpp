#include "network.hpp"
#include <cmath>
#include <random>
#include <iterator>
#include <thread>
#include <iostream>

namespace ml
{

std::ostream& operator<<(std::ostream& stream, const topology& topo)
{
    for (auto x : topo.layers)
    {
        stream << x << ',';
    }
    stream << std::endl;
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const neural_network& network)
{
    stream << network.m_topo;
    for (auto& pair : network.m_labels)
    {
        stream << pair.first << ",";
    }
    stream << std::endl;
    for (int i = 0; i < network.m_weights.size(); i++)
    {
        stream << network.m_weights[i] << std::endl;
        stream << network.m_biases[i] << std::endl;
    }
    return stream;
}

void neural_network::hadamard_product(VectorXd& lhs, const VectorXd& rhs)
{
    for (int i = 0; i < lhs.size(); i++)
    {
        lhs(i) *= rhs(i);
    }
}

std::map<std::string, VectorXd> neural_network::generate_class_vector_map(std::set<std::string> classes)
{
    std::map<std::string, VectorXd> ret_map;

    auto classes_iter = classes.begin();
    for (int i = 0; i < classes.size(); i++)
    {
        // output vectors for a multiclass classifier are as long as there are unique classes
        VectorXd tmp_vec(classes.size());
        tmp_vec.setZero(tmp_vec.rows(), tmp_vec.cols());
        tmp_vec(i) = 1.0;

        ret_map[*classes_iter] = tmp_vec;
        classes_iter++;
    }
    return ret_map;
}

neural_network::neural_network(topology topo, std::set<std::string> labels) : m_topo(topo), m_labels{generate_class_vector_map(labels)}
{
    for (int i = 1; i < m_topo.layers.size(); i++)
    {
        // populate m_weights with empty matricies matching topology
        m_weights.emplace_back(m_topo.layers[i], m_topo.layers[i - 1]);
        m_latest_weight_gradients.emplace_back(m_topo.layers[i], m_topo.layers[i - 1]);
        // populate m_biases with empty vectors matching topology
        m_biases.emplace_back(m_topo.layers[i]);
        m_latest_bias_gradients.emplace_back(m_topo.layers[i]);
    }
    // randomize weights and biases
    randomize();
}

neural_network::neural_network(std::vector<MatrixXd> weights, std::vector<VectorXd> biases, std::set<std::string> labels) : m_weights{weights}, m_biases{biases}, m_labels{generate_class_vector_map(labels)}
{
    m_latest_weight_gradients = m_weights;
    m_latest_bias_gradients   = m_biases;
    for (int i = 0; i < m_latest_weight_gradients.size(); i++)
    {
        m_latest_weight_gradients[i].setZero(m_latest_weight_gradients[i].rows(), m_latest_weight_gradients[i].cols());
        m_latest_bias_gradients[i].setZero(m_latest_bias_gradients[i].rows(), m_latest_bias_gradients[i].cols());
    }

    m_topo.layers.push_back(weights[0].cols());
    for (int i = 0; i < m_biases.size(); i++)
    {
        m_topo.layers.push_back(m_biases[i].size());
    }
}

void neural_network::randomize()
{
    std::random_device               rd;
    std::default_random_engine       generator(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // sets random weights
    for (auto& matrix : m_weights)
    {
        for (unsigned int i = 0; i < matrix.rows(); i++)
        {
            for (unsigned int j = 0; j < matrix.cols(); j++)
            {
                matrix(i, j) = dis(generator);
            }
        }
    }

    // sets random biases
    for (auto& vector : m_biases)
    {
        for (unsigned int i = 0; i < vector.size(); i++)
        {
            vector(i) = dis(generator);
        }
    }
}

void neural_network::logistic(VectorXd& vector)
{
    for (int i = 0; i < vector.size(); i++)
    {
        vector(i) = 1 / (1 + std::exp(-vector(i)));
    }
}

void neural_network::logistic_derivative(VectorXd& vector)
{
    for (int i = 0; i < vector.size(); i++)
    {
        vector(i) = (1 / (1 + std::exp(-vector(i)))) * (1 - (1 / (1 + std::exp(-vector(i)))));
    }
}

std::pair<std::string, double> neural_network::classify(const VectorXd& input_activations) const
{
    VectorXd current_activations = input_activations;

    // normalize input vector to range [0,1]
    double max = 0;
    for (int i = 0; i < current_activations.size(); i++)
    {
        if (current_activations(i) > max)
        {
            max = current_activations(i);
        }
    }
    if (max != 0)
    {
        for (int i = 0; i < current_activations.size(); i++)
        {
            current_activations(i) = current_activations(i) / max;
        }
    }

    // forward propogate inputs through network
    for (int i = 0; i < m_biases.size(); i++)
    {
        current_activations = m_weights[i] * current_activations + m_biases[i];
        logistic(current_activations);
    }

    // Find the largest activation and return it's value and associated class
    double largest_activation = 0;
    int    largest_index      = 0;
    for (int i = 0; i < current_activations.size(); i++)
    {
        if (current_activations[i] > largest_activation)
        {
            largest_index      = i;
            largest_activation = current_activations[i];
        }
    }

    // Find which label this index corresponds to
    auto map_iter = m_labels.begin();
    std::advance(map_iter, largest_index);
    return {map_iter->first, largest_activation};
}

std::pair<std::vector<std::string>, MatrixXi> neural_network::generate_confusion_matrix(const std::vector<std::pair<VectorXd, std::string>>& test_data) const
{
    // square matrix of size m_labels.size() x m_labels.size()
    MatrixXi                 ret_mat(m_labels.size(), m_labels.size());
    std::vector<std::string> ret_vec;
    ret_mat.setZero(ret_mat.rows(), ret_mat.cols());

    for (auto label : m_labels)
    {
        ret_vec.push_back(label.first);
    }

    // Using iterator differences to index into matrix, this is needed because std::map iterators are not random access
    // This works because m_labels is never modified after construction, and is kept in order due to std::map
    for (const auto& record : test_data)
    {
        auto result          = classify(record.first);
        int  predicted_index = std::distance(m_labels.begin(), m_labels.find(result.first));
        int  actual_index    = std::distance(m_labels.begin(), m_labels.find(record.second));
        ret_mat(actual_index, predicted_index)++;
    }

    return {ret_vec, ret_mat};
}

std::vector<std::pair<VectorXd, VectorXd>> neural_network::forward_propagate_return_all(const VectorXd& input_activations, const std::vector<MatrixXd>& local_weights, const std::vector<VectorXd>& local_biases) const
{
    // ret_vec stores the calculated activations and weighted sums
    // is in the form [[a-0,z-0] ... [a-L,z-L]] where a represents the activation vector
    // and z is the weighted sum vector, L is the number of layers.
    std::vector<std::pair<VectorXd, VectorXd>> ret_vec{};
    ret_vec.reserve(local_biases.size() + 1);
    ret_vec.emplace_back(input_activations, VectorXd(input_activations.size()));
    for (int i = 0; i < local_biases.size(); i++)
    {
        // vec = z vector
        VectorXd                      vec = local_weights[i] * ret_vec[i].first + local_biases[i];
        std::pair<VectorXd, VectorXd> tmp_pair;
        tmp_pair.second = vec;
        logistic(vec);
        // vec = a vector
        tmp_pair.first = vec;
        ret_vec.push_back(tmp_pair);
    }
    return ret_vec;
}

double neural_network::back_propagate(const std::vector<std::pair<VectorXd, VectorXd>>& training_data, const std::span<int>& mini_batch, double learning_rate)
{
    // training_data is in the form [[traning input data, expected output] ... ]

    // copy shared data to avoid data races
    std::unique_lock lock(mtx);
    auto             local_topo    = m_topo;
    auto             local_weights = m_weights;
    auto             local_biases  = m_biases;
    lock.unlock();

    // prepopulate weight_gradients and bias_gradients with matrix/vecotrs of same dimension as m_weights and m_biases
    std::vector<MatrixXd> weight_gradients;
    std::vector<VectorXd> bias_gradients;
    for (int i = 1; i < local_topo.layers.size(); i++)
    {
        weight_gradients.emplace_back(local_topo.layers[i], local_topo.layers[i - 1]);
        bias_gradients.emplace_back(local_topo.layers[i]);
        weight_gradients[i - 1].setZero(weight_gradients[i - 1].rows(), weight_gradients[i - 1].cols());
        // maybe replace with zero_vector to avoid clear
        bias_gradients[i - 1].setZero(bias_gradients[i - 1].rows(), bias_gradients[i - 1].cols());
    }

    int    mini_batch_size = mini_batch.size();
    double cost            = 0;

    for (int i = 0; i < mini_batch_size; i++)
    {
        // calculate weighted sums and activations of all layers in the network
        // for each input in the training set
        auto     input_results = forward_propagate_return_all(training_data[mini_batch[i]].first, local_weights, local_biases);
        int      current_layer = input_results.size() - 1;
        // first calculate gradient of cost function with respect to last activation layer. C = 0.5(a - y)^2
        VectorXd layer_errors  = input_results[current_layer].first - training_data[mini_batch[i]].second;
        for (int k = 0; k < layer_errors.size(); k++)
        {
            cost += 0.5 * layer_errors(k) * layer_errors(k);
        }
        logistic_derivative(input_results[current_layer].second);
        hadamard_product(layer_errors, input_results[current_layer].second);

        // input_results has one more layer than bias_gradients and weight_gradients. Therefore after
        // current_layer -= 1, current_layer refers to last layer of bias/weight_gradients and the second
        // last layer of input_results
        for (current_layer -= 1; current_layer >= 0; current_layer--)
        {
            bias_gradients[current_layer] += layer_errors;
            weight_gradients[current_layer] += layer_errors * input_results[current_layer].first.transpose();

            // derivative of layer weighted sums
            logistic_derivative(input_results[current_layer].second);
            // product of weight matrix and layer errors from next layer
            layer_errors = local_weights[current_layer].transpose() * layer_errors;
            // element-wise multiplication by activation-function-derivative of weighted sums
            hadamard_product(layer_errors, input_results[current_layer].second);
        }
    }

    // average gradients over training set size and update weights and biases
    lock.lock();
    for (int i = 0; i < m_weights.size(); i++)
    {
        // This incorporation of the previous weights and biases is known as momentum, it accelerates training. The coefficient of
        /// 0.9 is hardcoded for now but could be made dynamic in the future
        m_latest_weight_gradients[i] = 0.9 * m_latest_weight_gradients[i] + learning_rate * weight_gradients[i] / mini_batch_size;
        m_latest_bias_gradients[i]   = 0.9 * m_latest_bias_gradients[i] + learning_rate * bias_gradients[i] / mini_batch_size;

        m_weights[i] -= m_latest_weight_gradients[i];
        m_biases[i] -= m_latest_bias_gradients[i];
    }

    return cost / mini_batch_size;
}

void neural_network::train(const std::vector<std::pair<VectorXd, std::string>>& training_data, bool print_cost, std::atomic_bool& stopped, double learning_rate, std::size_t mini_batch_size)
{
    // normalize training data to the range [0,1]
    std::vector<std::pair<VectorXd, VectorXd>> normalized_training_data;
    normalized_training_data.reserve(training_data.size());
    for (auto record : training_data)
    {
        double max = 0;
        for (int i = 0; i < record.first.size(); i++)
        {
            if (record.first(i) > max)
            {
                max = record.first(i);
            }
        }
        if (max != 0)
        {
            for (int i = 0; i < record.first.size(); i++)
            {
                record.first(i) = record.first(i) / max;
            }
        }
        normalized_training_data.push_back({record.first, m_labels.at(record.second)});
    }

    std::random_device rd;
    std::mt19937       gen(rd());
    std::vector<int>   data_indices;
    // Initialize a vector of indecies into normalized_training_data, this is used later to create mini_batches
    for (int i = 0; i < normalized_training_data.size(); i++)
    {
        data_indices.push_back(i);
    }

    for (int i = 0; i < m_latest_weight_gradients.size(); i++)
    {
        m_latest_weight_gradients[i].setZero(m_latest_weight_gradients[i].rows(), m_latest_weight_gradients[i].cols());
        m_latest_bias_gradients[i].setZero(m_latest_bias_gradients[i].rows(), m_latest_bias_gradients[i].cols());
    }

    if (!mini_batch_size || mini_batch_size > training_data.size())
    {
        mini_batch_size = training_data.size();
    }

    while (!stopped)
    {
        double                   cost = 0;
        std::mutex               cost_mtx;
        std::vector<std::thread> threads;

        // Create a vector of randomized indecies into normalized_training_data
        std::vector<int> randomized_indices = data_indices;
        std::shuffle(randomized_indices.begin(), randomized_indices.end(), gen);

        int num_backprops = std::ceil(randomized_indices.size() / (double)mini_batch_size);
        int num_threads   = std::min({(int)std::thread::hardware_concurrency(), num_backprops});

        // Each thread roughly calculates num_backprops/num_threads back propogations,
        // Thread , will calculate every num_threads'th backprop starting with backprop n
        for (int i = 0; i < num_threads; i++)
        {
            auto func = [i, num_backprops, num_threads, learning_rate, mini_batch_size, &normalized_training_data, &randomized_indices, &cost, &cost_mtx, this]()
            {
                // back_propagate is passed a span into randomized_indices that indicate which items in normalized_training_data it should process,
                // this way I'm able to partition normalized_training_data randomlly without any expensive copying
                double local_cost       = 0;
                int    data_start_index = i * mini_batch_size;
                for (int j = i; j < num_backprops; j += num_threads)
                {
                    // handles the last backprop that might have to process more than mini_batch_size items
                    if (j == num_backprops - 1 && data_start_index != randomized_indices.size() - 1)
                    {
                        local_cost += back_propagate(normalized_training_data, {&randomized_indices[data_start_index], &randomized_indices[randomized_indices.size() - 1]}, learning_rate);
                    }
                    else // each backprop processes mini_batch_size training examples starting at the data_start_index index of randomized_indices
                    {
                        local_cost += back_propagate(normalized_training_data, {&randomized_indices[data_start_index], mini_batch_size}, learning_rate);
                    }
                    // increment data_start_index the size of a batch times the number of threads,
                    // this effectively allows each thread to process each num_threads'th backprop
                    data_start_index += num_threads * mini_batch_size;
                }

                std::unique_lock lock(cost_mtx);
                cost += local_cost;
            };

            threads.emplace_back(func);
        }

        // join backprop threads
        for (auto& thread : threads)
        {
            thread.join();
        }

        cost /= num_backprops;
        // A better implementation would to have a callback that can deal with cost
        // how the caller desires, this would eliminate the need for print_cost and cout
        // however this is fine for now.
        if (print_cost)
        {
            std::cout << cost << std::endl;
        }
    }
}

};