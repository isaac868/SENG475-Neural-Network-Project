#include "auxiliary_functions.hpp"

std::vector<std::string> input_csv(std::istream& stream)
{
    std::vector<std::string> ret_vec;
    std::string              tmp;
    std::getline(stream, tmp);

    if (!tmp.empty() && tmp[tmp.size() - 1] == '\r')
        tmp.erase(tmp.size() - 1);

    if (tmp.size() == 0)
    {
        return ret_vec;
    }

    int start = 0;
    int end   = 0;
    while (end != tmp.size())
    {
        end = tmp.find(',', start);
        if (end == std::string::npos)
        {
            end = tmp.size();
        }
        if (end - start)
        {
            ret_vec.push_back(tmp.substr(start, end - start));
        }
        start = end + 1;
    }
    return ret_vec;
}

MatrixXd readMatrix(std::istream& stream, int rows, int cols)
{
    int      temp_cols = 0, temp_rows = 0;
    MatrixXd result(rows, cols);

    while (!stream.eof() && temp_rows < rows)
    {
        std::string line;
        std::getline(stream, line);

        int               temp_cols = 0;
        std::stringstream str(line);
        while (!str.eof() && temp_cols < cols)
        {
            double tmp;
            str >> tmp;
            result(temp_rows, temp_cols) = tmp;
            temp_cols++;
        }

        temp_rows++;
    }

    return result;
}

void stream_network_data(std::vector<MatrixXd>& weights, std::vector<VectorXd>& biases, std::set<std::string>& labels, std::istream& stream)
{
    auto topo_vec  = input_csv(stream);
    auto label_vec = input_csv(stream);

    std::vector<int> topo_ints;

    try
    {
        int num_labels = std::stoi(topo_vec[topo_vec.size() - 1]);
        if (num_labels != label_vec.size())
        {
            throw std::runtime_error("Topology improperly formatted in trained network");
        }

        for (int i = 0; i < topo_vec.size(); i++)
        {
            topo_ints.push_back(std::stoi(topo_vec[i]));
        }
    }
    catch (const std::invalid_argument&)
    {
        throw std::runtime_error("Topology improperly formatted in trained network");
    }

    for (auto str : label_vec)
    {
        labels.insert(str);
    }
    for (int i = 0; i < topo_vec.size() - 1; i++)
    {
        MatrixXd tmp_weights = readMatrix(stream, topo_ints[i + 1], topo_ints[i]);
        weights.push_back(tmp_weights);
        VectorXd tmp_biases = readMatrix(stream, topo_ints[i + 1], 1);
        biases.push_back(tmp_biases);
    }
}