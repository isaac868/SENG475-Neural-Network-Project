#include "test_network.hpp"
#include <fstream>
#include <boost/numeric/ublas/io.hpp>
#include "boost/program_options.hpp"
#include <filesystem>

namespace po = boost::program_options;
namespace ub = boost::numeric::ublas;

void stream_test_data(int input_size, std::vector<VectorXd>& test_inputs, std::vector<std::string>& test_outputs, std::istream& stream)
{
    try
    {
        int                      row_size = -1;
        std::vector<std::string> str_test_data;
        do
        {
            str_test_data = input_csv(stream);
            if (!str_test_data.size())
            {
                break;
            }

            // Ensure test data rows are all the same length. Also ensures that the row size is either the same as
            // the network first layer size or one more than it (to fascilitate confusion network generation)
            if ((input_size != str_test_data.size() - 1 && input_size != str_test_data.size()) || str_test_data.size() != row_size && row_size != -1)
            {
                throw std::runtime_error("Test data improperly formatted");
            }
            row_size = str_test_data.size();

            VectorXd tmp_vec(input_size);
            for (int i = 0; i < input_size; i++)
            {
                tmp_vec(i) = std::stod(str_test_data[i]);
            }
            test_inputs.push_back(tmp_vec);

            if (str_test_data.size() > input_size)
            {
                test_outputs.push_back(str_test_data[input_size]);
            }
        }
        while ((str_test_data.size() != 0) && !stream.eof() && !stream.fail());
    }
    catch (const std::invalid_argument&)
    {
        throw std::runtime_error("Test data improperly formatted");
    }
};

void output_results(const std::vector<VectorXd>& test_inputs, const std::vector<std::string>& test_outputs, const ml::neural_network& nn, std::ostream& stream)
{

    if (test_outputs.size() > 0) // labels provided, generate confusion matrix given expected outputs
    {
        std::vector<std::pair<VectorXd, std::string>> tmp_vec;
        for (int i = 0; i < test_outputs.size(); i++)
        {
            tmp_vec.push_back({test_inputs[i], test_outputs[i]});
        }
        auto confusion_matrix = nn.generate_confusion_matrix(tmp_vec);

        // count total predictions and correct predictions (along the diagonal)
        double total_predictions   = 0;
        double correct_predictions = 0;
        for (int i = 0; i < confusion_matrix.second.rows(); i++)
        {
            for (int j = 0; j < confusion_matrix.second.cols(); j++)
            {
                total_predictions += confusion_matrix.second(i, j);
                if (i == j)
                {
                    correct_predictions += confusion_matrix.second(i, j);
                }
            }
        }

        // print model accuracy in top left
        stream << 100 * correct_predictions / total_predictions;
        // print column headers
        for (auto label : confusion_matrix.first)
        {
            stream << ',' << label;
        }

        for (int i = 0; i < confusion_matrix.first.size(); i++)
        {
            stream << std::endl;
            stream << confusion_matrix.first[i];
            for (int j = 0; j < confusion_matrix.second.cols(); j++)
            {
                stream << ',' << confusion_matrix.second(i, j);
            }
        }
    }
    else // labels not provided, output predictions and confidence
    {
        stream << "Input";
        for (int i = 0; i < test_inputs[0].size(); i++)
        {
            stream << ',';
        }
        stream << "Prediction,Confidence(%)";
        for (const auto& input : test_inputs)
        {
            stream << std::endl;
            auto result = nn.classify(input);
            for (int i = 0; i < input.size(); i++)
            {
                stream << input(i) << ',';
            }
            stream << result.first << ',';
            stream << result.second * 100;
        }
    }
    stream << std::endl;
}

int main(int ac, char** av)
{
    po::options_description desc("Allowed options");
    // clang-format off
        desc.add_options()  ("help", "display help message")
                            ("test-data,d", po::value<std::string>(), "String specifying a file to use for test data. If unspecified test data will be read from standard input.")
                            ("trained-network,n", po::value<std::string>(), "String specifying a file to read network weights and biases from. If unspecified, network data will be read from standard input.")
                            ("test-results,r", po::value<std::string>(), "String specifying a file to write test results to. If unspecified, test results will be written to standard output");
    // clang-format on

    try
    {
        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            std::cout << desc << "\n";
            std::cout << "Note: One of either test-data or network-file must be specified.\n";
            return 1;
        }

        if (!vm.count("test-data") && !vm.count("trained-network"))
        {
            throw boost::program_options::error("One of either test-data or network-file must be specified");
        }

        std::vector<MatrixXd> weights;
        std::vector<VectorXd> biases;
        std::set<std::string> labels;

        if (vm.count("trained-network"))
        {
            std::string file_name = vm["trained-network"].as<std::string>();
            if (!std::filesystem::exists(file_name))
            {
                throw std::runtime_error(file_name + ": No such file");
            }
            std::ifstream fstream;
            fstream.open(file_name);
            stream_network_data(weights, biases, labels, fstream);
            fstream.close();
        }
        else // network data is read from standard input
        {
            stream_network_data(weights, biases, labels, std::cin);
        }

        ml::neural_network nn(weights, biases, labels);

        std::vector<VectorXd>    test_inputs;
        std::vector<std::string> test_outputs;

        if (vm.count("test-data"))
        {
            std::string file_name = vm["test-data"].as<std::string>();
            if (!std::filesystem::exists(file_name))
            {
                throw std::runtime_error(file_name + ": No such file");
            }
            std::ifstream fstream;
            fstream.open(vm["test-data"].as<std::string>());
            stream_test_data(weights[0].cols(), test_inputs, test_outputs, fstream);
            fstream.close();
        }
        else // network data is read from standard input
        {
            stream_test_data(weights[0].cols(), test_inputs, test_outputs, std::cin);
        }

        // Check test data labels are in the pretrained network
        for (auto& str : test_outputs)
        {
            if (labels.find(str) == labels.end())
            {
                throw std::runtime_error("Test data incompatible with network");
            }
        }

        if (vm.count("test-results"))
        {
            std::ofstream fstream;
            fstream.open(vm["test-results"].as<std::string>());
            output_results(test_inputs, test_outputs, nn, fstream);
            fstream.close();
        }
        else
        {
            output_results(test_inputs, test_outputs, nn, std::cout);
        }
    }
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