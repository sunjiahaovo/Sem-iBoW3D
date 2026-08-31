#include "Data_IO.h"

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

        float parse_float_or_throw(const string& token,
                                   const string& path,
                                   int row,
                                   int col)
        {
            char* end = nullptr;
            errno = 0;
            const char* begin = token.c_str();
            float value = strtof(begin, &end);
            if(begin == end || errno == ERANGE)
            {
                throw runtime_error("Invalid numeric value at " + location(path, row, col) + ": " + token);
            }
            while(*end == ' ' || *end == '\t' || *end == '\r')
            {
                end++;
            }
            if(*end != '\0' || !isfinite(value))
            {
                throw runtime_error("Invalid numeric value at " + location(path, row, col) + ": " + token);
            }
            return value;
        }

        int parse_int_or_throw(const string& token,
                               const string& path,
                               int row)
        {
            char* end = nullptr;
            errno = 0;
            const char* begin = token.c_str();
            long value = strtol(begin, &end, 10);
            if(begin == end || errno == ERANGE)
            {
                throw runtime_error("Invalid loop id at " + path + ":" + to_string(row) + ": " + token);
            }
            while(*end == ' ' || *end == '\t' || *end == '\r')
            {
                end++;
            }
            if(*end != '\0' || value < numeric_limits<int>::min() || value > numeric_limits<int>::max())
            {
                throw runtime_error("Invalid loop id at " + path + ":" + to_string(row) + ": " + token);
            }
            return static_cast<int>(value);
        }
    }

    // function to read data from KITTI or Complex Urban
    vector<float> read_lidar_data_KITTI_CU(const std::string lidar_data_path)
    {
        std::ifstream lidar_data_file;
        lidar_data_file.open(lidar_data_path, std::ifstream::in | std::ifstream::binary);
        if(!lidar_data_file)
        {
            throw runtime_error("Failed to open lidar data file: " + lidar_data_path);
        }

        lidar_data_file.seekg(0, std::ios::end);
        const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
        lidar_data_file.seekg(0, std::ios::beg);

        std::vector<float> lidar_data_buffer(num_elements);
        lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));
        return lidar_data_buffer;
    }


    // function to read data from NCLT
    vector<short> read_lidar_data_NCLT(const std::string lidar_data_path)
    {
        std::ifstream lidar_data_file;
        lidar_data_file.open(lidar_data_path, std::ifstream::in | std::ifstream::binary);
        if(!lidar_data_file)
        {
            throw runtime_error("Failed to open lidar data file: " + lidar_data_path);
        }

        lidar_data_file.seekg(0, std::ios::end);
        const size_t num_elements = lidar_data_file.tellg() / sizeof(short);
        lidar_data_file.seekg(0, std::ios::beg);

        std::vector<short> lidar_data_buffer(num_elements);
        lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(short));
        return lidar_data_buffer;
    }


    vector<vector<float>> read_ground_truth_pose(const std::string pose_data_path)
    {
        vector<vector<float>> gt_pose;
        ifstream fp1(pose_data_path);
        if(!fp1.is_open())
        {
            throw runtime_error("Failed to open ground-truth pose file: " + pose_data_path);
        }
        string line1;
        int rowNum = 0;
        while (getline(fp1,line1)){ // read each line
            rowNum++;
            vector<float> data_line;
            string number;
            istringstream readstr(line1); // convert string to stream
            // data in one line are splitted by ","
            for(int j = 0;j < 3;j++){  // correspond to the number of data in each line
                if(!getline(readstr,number,',')) // get data
                {
                    throw runtime_error("Expected at least 3 pose values at " + pose_data_path + ":" +
                                        to_string(rowNum));
                }
                data_line.push_back(parse_float_or_throw(number, pose_data_path, rowNum, j + 1));
            }
            gt_pose.push_back(data_line); // insert in the result vector
        }
        return gt_pose;
    }

    vector<int> read_loop_ID(const std::string loopID_data_path)
    {
        vector<int> frame_loop;
        ifstream fp2(loopID_data_path);
        if(!fp2.is_open())
        {
            throw runtime_error("Failed to open loop id file: " + loopID_data_path);
        }
        string line2;
        int rowNum = 0;
        while (getline(fp2,line2)){ // read each line
            rowNum++;
            string number = line2;
            size_t comma_pos = number.find(',');
            if(comma_pos != string::npos)
            {
                string tail = number.substr(comma_pos + 1);
                if(tail.find_first_not_of(" \t\r") != string::npos)
                {
                    throw runtime_error("Unexpected trailing token in loop id file at " +
                                        loopID_data_path + ":" + to_string(rowNum));
                }
                number = number.substr(0, comma_pos);
            }
            int frameID = parse_int_or_throw(number, loopID_data_path, rowNum);
            frame_loop.push_back(frameID); // insert in the result vector
        }

        return frame_loop;
    }

    void output_loopList(const std::string output_path, vector<pair<int, int>> loopList)
    {
        ofstream outfile;
        outfile.open(output_path);
        if (outfile.is_open()) {
            for (auto& p : loopList) {
                outfile << p.first << ", " << p.second << endl;
            }
            outfile.close();
        }
    }
}
