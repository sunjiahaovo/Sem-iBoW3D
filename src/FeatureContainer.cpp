#include "FeatureContainer.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

using namespace std;

namespace iBoW3D
{
    namespace
    {
        string location(const string& path, int row, int col)
        {
            return path + ":" + to_string(row) + ":" + to_string(col);
        }

        float parse_float_token_or_throw(const string& token,
                                         const string& path,
                                         int row,
                                         int col)
        {
            char* end = nullptr;
            errno = 0;
            const char* begin = token.c_str();
            float value = strtof(begin, &end);
            if(begin == end || *end != '\0' || errno == ERANGE || !isfinite(value))
            {
                throw runtime_error("Invalid descriptor value at " + location(path, row, col) + ": " + token);
            }
            return value;
        }

        void read_descriptor_file_or_throw(const string& path,
                                           const string& description,
                                           int expected_rows,
                                           int feature_dim,
                                           cv::Mat& output)
        {
            ifstream fp(path);
            if(!fp.is_open())
            {
                throw runtime_error("Failed to open " + description + " descriptor file: " + path);
            }

            string line;
            int rowNum = 0;
            while(getline(fp, line))
            {
                if(rowNum >= expected_rows)
                {
                    throw runtime_error("Too many rows in " + description + " descriptor file: " + path);
                }

                istringstream readstr(line);
                vector<string> tokens;
                string token;
                while(readstr >> token)
                {
                    tokens.push_back(token);
                }

                if(static_cast<int>(tokens.size()) != feature_dim)
                {
                    throw runtime_error("Expected " + to_string(feature_dim) + " descriptor values but found " +
                                        to_string(tokens.size()) + " at " + path + ":" + to_string(rowNum + 1));
                }

                for(int j = 0; j < feature_dim; j++)
                {
                    output.at<float>(rowNum, j) =
                        parse_float_token_or_throw(tokens[j], path, rowNum + 1, j + 1);
                }
                rowNum++;
            }

            if(rowNum != expected_rows)
            {
                throw runtime_error("Unexpected row count in " + description + " descriptor file: " + path +
                                    "; expected " + to_string(expected_rows) +
                                    ", found " + to_string(rowNum));
            }
        }
    }

    FeatureContainer::FeatureContainer(int keypoint_num_, int feature_dim_, int pcd_num_, int frameID_, const std::string key_feature_path_, const std::string all_feature_path_):
    keypoint_num(keypoint_num_), feature_dim(feature_dim_), pcd_num(pcd_num_), frameID(frameID_), key_feature_path(key_feature_path_), all_feature_path(all_feature_path_)
    {
        key_feat.create(keypoint_num, feature_dim, CV_32F);
        all_feat.create(pcd_num, feature_dim, CV_32F);

        key_feat = cv::Mat::zeros(keypoint_num, feature_dim, CV_32F);
        all_feat = cv::Mat::zeros(pcd_num, feature_dim, CV_32F);

        // key_feat.resize(keypoint_num, feature_dim);
        // all_feat.resize(pcd_num, feature_dim);

        // read key features
        const string key_file = key_feature_path+"descriptors_"+to_string(frameID)+".txt";
        read_descriptor_file_or_throw(key_file, "key", keypoint_num, feature_dim, key_feat);

        // read all features
        const string all_file = all_feature_path+"descriptors_"+to_string(frameID)+".txt";
        read_descriptor_file_or_throw(all_file, "all-point", pcd_num, feature_dim, all_feat);

        // // convert double to float
        // key_feat.convertTo(key_feat, CV_32F);
        // all_feat.convertTo(all_feat, CV_32F);
    }

    void FeatureContainer::update(int pcd_num_, int frameID_)
    {
        pcd_num = pcd_num_;
        frameID = frameID_;

        key_feat.create(keypoint_num, feature_dim, CV_32F);
        all_feat.create(pcd_num, feature_dim, CV_32F);

        key_feat = cv::Mat::zeros(keypoint_num, feature_dim, CV_32F);
        all_feat = cv::Mat::zeros(pcd_num, feature_dim, CV_32F);

        // key_feat.resize(keypoint_num, feature_dim);
        // all_feat.resize(pcd_num, feature_dim);

        // read key features
        const string key_file = key_feature_path+"descriptors_"+to_string(frameID)+".txt";
        read_descriptor_file_or_throw(key_file, "key", keypoint_num, feature_dim, key_feat);

        // read all features
        const string all_file = all_feature_path+"descriptors_"+to_string(frameID)+".txt";
        read_descriptor_file_or_throw(all_file, "all-point", pcd_num, feature_dim, all_feat);

        // // convert double to float
        // key_feat.convertTo(key_feat, CV_32F);
        // all_feat.convertTo(all_feat, CV_32F);
    }

}

/* Below codes are for test */

// #include "Data_IO.h"

// int main()
// {

//     string dataset = "KITTI";
//     string seq = "00";


//     string dataset_folder;
//     dataset_folder = "./data/"+dataset+"/"+seq+"/velodyne/";

//     int frameID = 0;

//     stringstream lidar_data_path;
//     lidar_data_path << dataset_folder << setfill('0') << setw(6) << frameID << ".bin";
//     cout << lidar_data_path.str() << endl;

//     vector<float> lidar_data = iBoW3D::read_lidar_data_KITTI_CU(lidar_data_path.str());

//     // pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>());
//     cout << "number of points: " << lidar_data.size()/4 << endl;

//     // for(std::size_t i = 0; i < lidar_data.size(); i += 4)
//     // {
//     //     pcl::PointXYZ point;
//     //     point.x = lidar_data[i];
//     //     point.y = lidar_data[i + 1];
//     //     point.z = lidar_data[i + 2];

//     //     current_cloud->push_back(point);
//     // }

//     string key_feature_path = "./descriptor_txt/D3F/"+dataset+"/"+seq+"/";
//     string all_feature_path = "./feature_txt/D3F/"+dataset+"/"+seq+"/";


//     iBoW3D::FeatureContainer test_FC(20, 32, lidar_data.size()/4, frameID, key_feature_path, all_feature_path);
//     cv::Mat text_kf = test_FC.getKeyFeature();
//     // Eigen::MatrixXf text_kf = test_FC.getKeyFeature();
//     cout << text_kf << endl;
//     cout << text_kf.size() << endl;
//     cout << text_kf.rows << endl;
//     cout << text_kf.cols << endl;
//     return 0;
// }
