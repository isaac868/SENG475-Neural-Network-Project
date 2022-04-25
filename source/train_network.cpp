#include "train_network.hpp"
#include "boost/program_options.hpp"
#include <fstream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <charconv>
#include <iostream>

namespace po = boost::program_options;
using namespace std::chrono_literals;

namespace ml
{

std::istream& operator>>(std::istream& stream, ml::topology& topo)
{
    try
    {
        auto         topo_str_vec = input_csv(stream);
        ml::topology ret_topo;

        for (auto& str : topo_str_vec)
        {
            int num = std::stoi(str);
            if (num < 1)
            {
                throw boost::program_options::error("Layers must have at least 1 node");
            }
            ret_topo.layers.push_back(num);
        }

        topo = ret_topo;
    }
    catch (const std::invalid_argument&)
    {
        throw boost::program_options::error("Topology improperly formatted");
    }
    return stream;
}

}

void validate_topology_requirements(int first_layer_size, int last_layer_size, const std::vector<std::pair<VectorXd, std::string>>& training_data)
{
    std::set<std::string> unique_labels;
    for (const auto& pair : training_data)
    {
        unique_labels.insert(pair.second);
        if (first_layer_size != pair.first.size())
        {
            throw std::runtime_error("Training input data does not match topology");
        }
    }
    if (unique_labels.size() != last_layer_size)
    {
        throw std::runtime_error("Training data labels do not match topology");
    }
}

void stream_training_data(std::vector<std::pair<VectorXd, std::string>>& training_data, std::istream& stream)
{
    try
    {
        int         row_size = -1;
        std::string str_test_data;
        do
        {
            std::vector<std::string> ret_vec;
            std::getline(stream, str_test_data);

            // Break when stream returns empty string
            if (!str_test_data.size())
            {
                break;
            }

            int num = std::count(str_test_data.begin(), str_test_data.end(), ',');
            if (str_test_data[str_test_data.size() - 1] != ',')
            {
                num++;
            }

            // Ensure training data rows are all the same length
            if (num != row_size && row_size != -1)
            {
                throw std::runtime_error("Training data improperly formatted");
            }
            row_size = num;

            VectorXd    tmp_vec(num - 1);
            const char* ptr = str_test_data.c_str();
            for (int i = 0; i < num - 1; i++)
            {
                double val = 0;
                auto   res = std::from_chars(ptr, str_test_data.c_str() + str_test_data.size(), val);
                if (res.ec == std::errc::result_out_of_range)
                {
                    throw std::runtime_error("asd");
                }
                else
                {
                    tmp_vec(i) = val;
                    ptr        = res.ptr + 1;
                }
            }
            training_data.emplace_back(std::move(tmp_vec), std::string(ptr, str_test_data.c_str() + str_test_data.size()));
        }
        while ((str_test_data.size() != 0) && !stream.eof() && !stream.fail());
    }
    catch (const std::invalid_argument&)
    {
        throw std::runtime_error("Training data improperly formatted");
    }
}

int main(int ac, char** av)
{
    po::options_description desc("Allowed options");
    // clang-format off
    desc.add_options()  ("help", "display help message")
                        ("timeout,t", po::value<int>(), "Integer number of seconds before training terminates, default is 5s.")
                        ("training-data,d", po::value<std::string>(), "String specifying a file to use for training data. If unspecified training data will be read from standard input.")
                        ("topology,T", po::value<ml::topology>(), "String specifying network topology. Incompatible with import-network-file.")
                        ("learning-rate,r", po::value<double>(), "Real number used for network learning rate. Default is 0.1")
                        ("batch-size,b", po::value<int>(), "Integer number used for gradient descent batch size. Value of 0 indicates the whole set should be processed each batch.")
                        ("import-network-file,i", po::value<std::string>(), "String specifying a file to read network weights and biases from. This can be used to seed the network from previous training.")
                        ("trained-network,n", po::value<std::string>(), "String specifying a file to write network weights and biases to. If unspecified, network data will be written to standard output.");
    // clang-format on

    try
    {
        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            std::cout << desc << "\n";
            std::cout << "Note: One of either topology or import-network-file must be specified.\n";
            return 1;
        }

        std::vector<std::pair<VectorXd, std::string>> training_data;
        auto                                          begin = std::chrono::system_clock::now();

        if (vm.count("training-data"))
        {
            std::string file_name = vm["training-data"].as<std::string>();
            if (!std::filesystem::exists(file_name))
            {
                throw std::runtime_error(file_name + ": No such file");
            }
            std::ifstream fstream;
            fstream.open(file_name);
            stream_training_data(training_data, fstream);
            fstream.close();
        }
        else // training data is read from standard input
        {
            stream_training_data(training_data, std::cin);
        }

        std::set<std::string> labels;
        for (const auto& record : training_data)
        {
            labels.insert(record.second);
        }

        if (!vm.count("topology") && !vm.count("import-network-file"))
        {
            throw boost::program_options::error("One of either topology or import-network-file must be specified");
        }

        double learning_rate = 0.1;
        int    batch_size    = 0;

        if (vm.count("learning-rate"))
        {
            learning_rate = vm["learning-rate"].as<double>();
        }

        if (vm.count("batch-size"))
        {
            batch_size = vm["batch-size"].as<int>();
        }

        // Immediately invoked lambda to create neural network.
        ml::neural_network nn = [&]()
        {
            // Initialize network with topology from command line, this superceeds import-network-file if both are specefied.
            if (vm.count("topology"))
            {
                ml::topology topo = vm["topology"].as<ml::topology>();
                if (topo.layers.size() < 2)
                {
                    throw boost::program_options::error("A minimum of two layers is required");
                }
                validate_topology_requirements(topo.layers[0], topo.layers[topo.layers.size() - 1], training_data);
                return ml::neural_network(topo, labels);
            }

            // else read network data from file
            std::string file_name = vm["import-network-file"].as<std::string>();
            if (!std::filesystem::exists(file_name))
            {
                throw std::runtime_error(file_name + " does not exist");
            }
            std::vector<MatrixXd> weights;
            std::vector<VectorXd> biases;
            std::set<std::string> labels_other;
            std::ifstream         fstream;

            // Imported network file cannot be read from standard input
            fstream.open(file_name);
            stream_network_data(weights, biases, labels_other, fstream);
            fstream.close();

            // Check training data labels are in the pretrained network
            for (auto& str : labels)
            {
                if (labels_other.find(str) == labels_other.end())
                {
                    throw std::runtime_error("Training data incompatible with network");
                }
            }

            validate_topology_requirements(weights[0].cols(), biases[biases.size() - 1].size(), training_data);
            return ml::neural_network(weights, biases, labels_other);
        }();

        int timeout = 5;
        if (vm.count("timeout"))
        {
            timeout = vm["timeout"].as<int>();
            if (timeout < 0)
            {
                throw boost::program_options::error("Timeout cannot be negative");
            }
        }

        // Training thread is created and time is monitored every 100ms to ensure timout parameter is observed.
        std::atomic_bool stop_training = false;
        std::thread      training_thread([&]() { nn.train(training_data, vm.count("trained-network"), stop_training, learning_rate, batch_size); });

        auto duration = std::chrono::system_clock::now() - begin;
        while (duration_cast<std::chrono::milliseconds>(duration).count() <= timeout * 1000)
        {
            std::this_thread::sleep_for(100ms);
            duration = std::chrono::system_clock::now() - begin;
        }
        stop_training = true;
        training_thread.join();

        if (vm.count("trained-network"))
        {
            std::ofstream fstream;
            fstream.open(vm["trained-network"].as<std::string>());
            fstream << nn;
            fstream.close();
        }
        else // trained network is written to standard output
        {
            std::cout << nn;
        }
    }
    // Required to catch this first as it contains additional information that is lost if I just catch boost::program_options::error
    catch (const boost::program_options::error_with_option_name& err)
    {
        std::cerr << err.what() << "\n";
        std::cerr << desc << "\n";
    }
    catch (const boost::program_options::error& err)
    {
        std::cerr << err.what() << "\n";
        std::cerr << desc << "\n";
    }
    catch (const std::runtime_error& err)
    {
        std::cerr << err.what() << "\n";
    }

    return 0;
}