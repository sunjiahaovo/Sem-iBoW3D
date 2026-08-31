#pragma once

#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <string>


#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


#include <math.h>

using namespace std;


namespace iBoW3D
{
    class LabelContainer
    {
        public:
            LabelContainer(int keypoint_num_, int frameID_, const std::string key_label_path_, int semantic_num_);

            ~LabelContainer(){}

            cv::Mat getKeyLabel(){return key_label;}
            // Eigen::MatrixXf getKeyFeature(){return key_label;}


        private:
            int keypoint_num;
            int frameID;
            int semantic_num;
            const std::string key_label_path;

            cv::Mat key_label;
            // Eigen::MatrixXd key_label;

    };
}
