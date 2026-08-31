#include "FeatureContainer.h"
#include "LabelContainer.h"
#include "Data_IO.h"
#include "iBoW3D.h"

#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace
{
void make_dir(const std::string& path)
{
    if(mkdir(path.c_str(), 0755) != 0 && errno != EEXIST)
    {
        throw std::runtime_error("Failed to create directory: " + path);
    }
}

void write_file(const std::string& path, const std::string& content)
{
    std::ofstream out(path);
    if(!out.is_open())
    {
        throw std::runtime_error("Failed to write file: " + path);
    }
    out << content;
}

template <typename Func>
void expect_throw(Func&& func, const std::string& needle)
{
    try
    {
        func();
    }
    catch(const std::exception& exc)
    {
        const std::string message = exc.what();
        if(message.find(needle) == std::string::npos)
        {
            throw std::runtime_error("Unexpected exception message: " + message);
        }
        return;
    }
    throw std::runtime_error("Expected exception containing: " + needle);
}

void expect_near(double actual, double expected, double tolerance, const std::string& label)
{
    if(std::fabs(actual - expected) > tolerance)
    {
        throw std::runtime_error(label + " expected " + std::to_string(expected) +
                                 " but got " + std::to_string(actual));
    }
}
}

int main(int argc, char** argv)
{
    const std::string root = argc > 1 ? argv[1] : "test_input_contracts_tmp";
    make_dir(root);
    const std::string key_dir = root + "/key/";
    const std::string all_dir = root + "/all/";
    const std::string label_dir = root + "/labels/";
    make_dir(key_dir);
    make_dir(all_dir);
    make_dir(label_dir);

    write_file(label_dir + "0.txt", "0\n-1\n13\n");
    iBoW3D::LabelContainer valid_labels(3, 0, label_dir, 14);
    if(valid_labels.getKeyLabel().at<int>(0, 0) != 0 ||
       valid_labels.getKeyLabel().at<int>(1, 0) != -1 ||
       valid_labels.getKeyLabel().at<int>(2, 0) != 13)
    {
        throw std::runtime_error("Valid labels were not loaded correctly");
    }

    write_file(label_dir + "1.txt", "0\n14\n1\n");
    expect_throw([&] { iBoW3D::LabelContainer labels(3, 1, label_dir, 14); },
                 "Semantic label out of range");

    write_file(label_dir + "2.txt", "0\nbad\n1\n");
    expect_throw([&] { iBoW3D::LabelContainer labels(3, 2, label_dir, 14); },
                 "Invalid semantic label");

    write_file(key_dir + "descriptors_0.txt", "1.5 2.25\n3.5 4.75\n");
    write_file(all_dir + "descriptors_0.txt", "0.1 0.2\n0.3 0.4\n0.5 0.6\n");
    iBoW3D::FeatureContainer valid_features(2, 2, 3, 0, key_dir, all_dir);
    expect_near(valid_features.getKeyFeature().at<float>(0, 0), 1.5, 1e-6, "key feature");
    expect_near(valid_features.getAllFeature().at<float>(2, 1), 0.6, 1e-6, "all feature");

    write_file(key_dir + "descriptors_1.txt", "1.0\n3.0 4.0\n");
    write_file(all_dir + "descriptors_1.txt", "0.1 0.2\n0.3 0.4\n0.5 0.6\n");
    expect_throw([&] { iBoW3D::FeatureContainer features(2, 2, 3, 1, key_dir, all_dir); },
                 "Expected 2 descriptor values but found 1");

    write_file(key_dir + "descriptors_2.txt", "1.0 2.0 3.0\n3.0 4.0\n");
    write_file(all_dir + "descriptors_2.txt", "0.1 0.2\n0.3 0.4\n0.5 0.6\n");
    expect_throw([&] { iBoW3D::FeatureContainer features(2, 2, 3, 2, key_dir, all_dir); },
                 "Expected 2 descriptor values but found 3");

    write_file(key_dir + "descriptors_3.txt", "1.0 nan\n3.0 4.0\n");
    write_file(all_dir + "descriptors_3.txt", "0.1 0.2\n0.3 0.4\n0.5 0.6\n");
    expect_throw([&] { iBoW3D::FeatureContainer features(2, 2, 3, 3, key_dir, all_dir); },
                 "Invalid descriptor value");

    std::vector<std::pair<double, int>> distances = {
        {0.25, 10},
        {0.75, 11},
        {10.0, 30},
    };
    std::vector<int> island;
    double mean = iBoW3D::iBoW3D::mean_candidate_distance(distances, 10, 2, &island);
    expect_near(mean, 0.5, 1e-12, "mean candidate distance");
    if(island.size() != 2 || island[0] != 10 || island[1] != 11)
    {
        throw std::runtime_error("Candidate island membership is wrong");
    }
    expect_throw([&] { iBoW3D::iBoW3D::mean_candidate_distance(distances, 10, 0, nullptr); },
                 "near_num must be positive");

    write_file(root + "/valid_poses.csv", "0,1,2\n3,4,5,ignored\n");
    auto poses = iBoW3D::read_ground_truth_pose(root + "/valid_poses.csv");
    if(poses.size() != 2 || poses[1].size() != 3)
    {
        throw std::runtime_error("Valid poses were not loaded correctly");
    }
    write_file(root + "/bad_poses.csv", "0,1\n");
    expect_throw([&] { iBoW3D::read_ground_truth_pose(root + "/bad_poses.csv"); },
                 "Expected at least 3 pose values");
    write_file(root + "/nan_poses.csv", "0,nan,2\n");
    expect_throw([&] { iBoW3D::read_ground_truth_pose(root + "/nan_poses.csv"); },
                 "Invalid numeric value");

    write_file(root + "/valid_loops.csv", "1\n2,\n");
    auto loops = iBoW3D::read_loop_ID(root + "/valid_loops.csv");
    if(loops.size() != 2 || loops[0] != 1 || loops[1] != 2)
    {
        throw std::runtime_error("Valid loop ids were not loaded correctly");
    }
    write_file(root + "/bad_loops.csv", "1,2\n");
    expect_throw([&] { iBoW3D::read_loop_ID(root + "/bad_loops.csv"); },
                 "Unexpected trailing token");
    write_file(root + "/invalid_loops.csv", "abc\n");
    expect_throw([&] { iBoW3D::read_loop_ID(root + "/invalid_loops.csv"); },
                 "Invalid loop id");

    std::cout << "input contract tests passed\n";
    return 0;
}
