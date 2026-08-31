#include "LabelContainer.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <stdexcept>

using namespace std;

namespace iBoW3D
{
    namespace
    {
        string location(const string& path, int row)
        {
            return path + ":" + to_string(row);
        }

        int parse_label_or_throw(const string& line,
                                 const string& path,
                                 int row,
                                 int semantic_num)
        {
            char* end = nullptr;
            errno = 0;
            const char* begin = line.c_str();
            long value = strtol(begin, &end, 10);
            if(begin == end || errno == ERANGE)
            {
                throw runtime_error("Invalid semantic label at " + location(path, row) + ": " + line);
            }
            while(*end == ' ' || *end == '\t' || *end == '\r')
            {
                end++;
            }
            if(*end != '\0')
            {
                throw runtime_error("Unexpected trailing token in semantic label at " + location(path, row) + ": " + line);
            }
            if(value < -1 || value >= semantic_num)
            {
                throw runtime_error("Semantic label out of range at " + location(path, row) +
                                    ": " + to_string(value) + " allowed range is -1 or [0, " +
                                    to_string(semantic_num - 1) + "]");
            }
            if(value < numeric_limits<int>::min() || value > numeric_limits<int>::max())
            {
                throw runtime_error("Semantic label cannot fit in int at " + location(path, row) + ": " + line);
            }
            return static_cast<int>(value);
        }
    }

    LabelContainer::LabelContainer(int keypoint_num_, int frameID_, const std::string key_label_path_, int semantic_num_):
    keypoint_num(keypoint_num_), frameID(frameID_), semantic_num(semantic_num_), key_label_path(key_label_path_)
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
            key_label.at<int>(rowNum, 0) = parse_label_or_throw(line, label_file, rowNum + 1, semantic_num);
            rowNum++;
        }
        if(rowNum != keypoint_num)
        {
            throw runtime_error("Unexpected row count in keypoint semantic label file: " + label_file);
        }

    }
}
