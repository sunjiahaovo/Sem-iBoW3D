#include "LabelContainer.h"

#include <stdexcept>

using namespace std;

namespace iBoW3D
{
    LabelContainer::LabelContainer(int keypoint_num_, int frameID_, const std::string key_label_path_):
    keypoint_num(keypoint_num_), frameID(frameID_), key_label_path(key_label_path_)
    {
        key_label.create(keypoint_num, 1, CV_32S);
        key_label = cv::Mat::zeros(keypoint_num, 1, CV_32S);

        // key_feat.resize(keypoint_num, 1);

        // read key labels which had been reset
        const string label_file = key_label_path+to_string(frameID)+".txt";
        ifstream fp(label_file);
        if(!fp.is_open())
        {
            throw runtime_error("Failed to open keypoint semantic label file: " + label_file);
        }
        string line;
        int rowNum = 0;
        while (getline(fp, line)){ // read each line
            if(rowNum >= keypoint_num)
            {
                throw runtime_error("Too many rows in keypoint semantic label file: " + label_file);
            }
            string number;
            istringstream readstr(line); // convert string to stream
            // data in one line are splitted by ","
            for(int j = 0; j < 1; j++){  // correspond to the number of data in each line
                getline(readstr, number, ' '); // get data
                key_label.at<int>(rowNum, j) = atof(number.c_str()); // convert string to int
            }
            rowNum++;
        }
        if(rowNum != keypoint_num)
        {
            throw runtime_error("Unexpected row count in keypoint semantic label file: " + label_file);
        }

    }
}
