#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <vector>

#include <Eigen/Dense>
#include <open3d/Open3D.h>

#include "DataBase.h"
#include "Data_IO.h"
#include "FeatureContainer.h"
#include "TimeSave.h"
#include "iBoW3D.h"

using namespace std;
using namespace open3d;

namespace
{
struct Options
{
    string dataset;
    string sequence;
    string project_root = ".";
    string scan_dir;
    string pose_file;
    string loop_file;
    string key_feature_dir;
    string all_feature_dir;
    string label_dir;
    string result_dir;
    string time_dir;

    int max_frames = 0;
    int keypoint_num = 20;
    int feature_dim = 32;
    int init_pcd_num = 400;
    int init_words_num = 50;
    int words_num_add = 10;
    int update_num = 200;
    int near_num = 8;
    int search_num = 5;
    int gap_num = 250;
    int max_iter = 1000;
    int ransac_n = 0;
    int semantic_num = 14;

    double lambda_word = 0.2;
    double fit_th = -1.0;
    double score_th = -1.0;
    double fit_th_2 = -1.0;
    double score_th_2 = -1.0;
    double check_th = 1000000.0;

    bool semantic = true;
    bool async_update = false;
    bool save_time = false;
    iBoW3D::RegistrationBackend registration_backend = iBoW3D::RegistrationBackend::RANSAC;
};

double seconds_between(const timeval& a, const timeval& b)
{
    return (b.tv_sec - a.tv_sec) + static_cast<double>(b.tv_usec - a.tv_usec) / 1000000.0;
}

bool file_exists(const string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool dir_exists(const string& path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

string trim_trailing_slash(string path)
{
    while(path.size() > 1 && path.back() == '/')
    {
        path.pop_back();
    }
    return path;
}

string ensure_trailing_slash(string path)
{
    if(!path.empty() && path.back() != '/')
    {
        path += "/";
    }
    return path;
}

string join_path(const string& lhs, const string& rhs)
{
    if(lhs.empty())
    {
        return rhs;
    }
    if(rhs.empty())
    {
        return lhs;
    }
    if(rhs.front() == '/')
    {
        return rhs;
    }
    if(lhs.back() == '/')
    {
        return lhs + rhs;
    }
    return lhs + "/" + rhs;
}

bool create_directories_if_needed(const string& path)
{
    if(path.empty())
    {
        return false;
    }

    string current;
    size_t pos = 0;
    if(path[0] == '/')
    {
        current = "/";
        pos = 1;
    }

    while(pos <= path.size())
    {
        size_t next = path.find('/', pos);
        string part = path.substr(pos, next == string::npos ? string::npos : next - pos);

        if(!part.empty())
        {
            if(!current.empty() && current.back() != '/')
            {
                current += "/";
            }
            current += part;

            struct stat st;
            if(stat(current.c_str(), &st) != 0)
            {
                if(mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
                {
                    return false;
                }
            }
            else if(!S_ISDIR(st.st_mode))
            {
                return false;
            }
        }

        if(next == string::npos)
        {
            break;
        }
        pos = next + 1;
    }

    return true;
}

string require_value(int& index, int argc, char** argv)
{
    if(index + 1 >= argc)
    {
        throw runtime_error(string("Missing value for ") + argv[index]);
    }
    index++;
    return argv[index];
}

int parse_int(const string& text, const string& name)
{
    size_t consumed = 0;
    int value = stoi(text, &consumed);
    if(consumed != text.size())
    {
        throw runtime_error("Invalid integer for " + name + ": " + text);
    }
    return value;
}

double parse_double(const string& text, const string& name)
{
    size_t consumed = 0;
    double value = stod(text, &consumed);
    if(consumed != text.size())
    {
        throw runtime_error("Invalid number for " + name + ": " + text);
    }
    return value;
}

void print_help(const char* program)
{
    cout
        << "Usage:\n"
        << "  " << program << " --dataset KITTI --sequence 05 --project-root /path/to/dataset_root [options]\n\n"
        << "Required:\n"
        << "  --dataset NAME                  Dataset name, e.g. KITTI, KITTI360, FC, CU, NCLT\n"
        << "  --sequence SEQ                  Sequence id, e.g. 00, 05, 09\n\n"
        << "Data layout options:\n"
        << "  --project-root DIR              Root containing data/, res_data/, descriptor_txt/, feature_txt/ (default: .)\n"
        << "  --scan-dir DIR                  Override point cloud directory; default: res_data/<dataset>/<seq>/D3F_allpoints\n"
        << "  --pose-file FILE                Override GTposes.csv path\n"
        << "  --loop-file FILE                Override loop_lst.csv path\n"
        << "  --key-feature-dir DIR           Override keypoint descriptor directory\n"
        << "  --all-feature-dir DIR           Override all-point descriptor directory\n"
        << "  --label-dir DIR                 Override keypoint semantic-label directory\n\n"
        << "Method options:\n"
        << "  --no-semantic                   Disable semantic-aided BoW and use geometric descriptors only\n"
        << "  --keypoint-num N                Keypoint descriptors per frame (default: 20)\n"
        << "  --feature-dim N                 Descriptor dimension (default: 32)\n"
        << "  --fit-th VALUE                  Final registration fitness threshold\n"
        << "  --score-th VALUE                Final registration RMSE threshold\n"
        << "  --fit-th2 VALUE                 Coarse island fitness threshold (default: --fit-th)\n"
        << "  --score-th2 VALUE               Coarse island RMSE threshold (default: --score-th)\n"
        << "  --search-num N                  Candidates kept per selected island (default: 5)\n"
        << "  --near-num N                    Frame-neighborhood width used to group candidate islands (default: 8)\n"
        << "  --gap-num N                     Recent-frame exclusion gap for loop candidates (default: 250)\n"
        << "  --max-iter N                    RANSAC/FGR max iterations (default: 1000)\n"
        << "  --ransac-n N                    Correspondences sampled by RANSAC, default 4 or 3 for CU\n"
        << "  --check-th VALUE                Range-check threshold (default: 1000000)\n"
        << "  --init-pcd-num N                Initial database size N_d (default: 400)\n"
        << "  --init-words-num N              Initial visual-word count (default: 50)\n"
        << "  --words-num-add N               Visual words added at each dictionary update (default: 10)\n"
        << "  --update-num N                  Dictionary update interval N_u (default: 200)\n"
        << "  --lambda-word VALUE             Coarse word-overlap threshold lambda_w (default: 0.2)\n"
        << "  --semantic-num N                Number of valid static semantic labels; valid labels are -1 or [0, N-1] (default: 14)\n"
        << "  --async-update                  Rebuild BoW dictionary in a background thread\n"
        << "  --registration-backend NAME     ransac or fgr (default: ransac)\n\n"
        << "Output options:\n"
        << "  --result-dir DIR                Directory for results.txt and looplist.txt\n"
        << "  --time-dir DIR                  Directory for timing text files; enables timing output\n"
        << "  --max-frames N                  Stop after N frames for smoke tests\n"
        << "  --help                          Show this help\n";
}

Options parse_args(int argc, char** argv)
{
    Options opt;
    for(int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        if(arg == "--help" || arg == "-h")
        {
            print_help(argv[0]);
            exit(0);
        }
        else if(arg == "--dataset") opt.dataset = require_value(i, argc, argv);
        else if(arg == "--sequence" || arg == "--seq") opt.sequence = require_value(i, argc, argv);
        else if(arg == "--project-root") opt.project_root = require_value(i, argc, argv);
        else if(arg == "--scan-dir") opt.scan_dir = require_value(i, argc, argv);
        else if(arg == "--pose-file") opt.pose_file = require_value(i, argc, argv);
        else if(arg == "--loop-file") opt.loop_file = require_value(i, argc, argv);
        else if(arg == "--key-feature-dir") opt.key_feature_dir = require_value(i, argc, argv);
        else if(arg == "--all-feature-dir") opt.all_feature_dir = require_value(i, argc, argv);
        else if(arg == "--label-dir") opt.label_dir = require_value(i, argc, argv);
        else if(arg == "--result-dir") opt.result_dir = require_value(i, argc, argv);
        else if(arg == "--time-dir")
        {
            opt.time_dir = require_value(i, argc, argv);
            opt.save_time = true;
        }
        else if(arg == "--max-frames") opt.max_frames = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--keypoint-num") opt.keypoint_num = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--feature-dim") opt.feature_dim = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--init-pcd-num") opt.init_pcd_num = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--init-words-num") opt.init_words_num = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--words-num-add") opt.words_num_add = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--update-num") opt.update_num = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--near-num") opt.near_num = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--search-num") opt.search_num = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--gap-num") opt.gap_num = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--max-iter") opt.max_iter = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--ransac-n") opt.ransac_n = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--semantic-num") opt.semantic_num = parse_int(require_value(i, argc, argv), arg);
        else if(arg == "--lambda-word") opt.lambda_word = parse_double(require_value(i, argc, argv), arg);
        else if(arg == "--fit-th") opt.fit_th = parse_double(require_value(i, argc, argv), arg);
        else if(arg == "--score-th") opt.score_th = parse_double(require_value(i, argc, argv), arg);
        else if(arg == "--fit-th2") opt.fit_th_2 = parse_double(require_value(i, argc, argv), arg);
        else if(arg == "--score-th2") opt.score_th_2 = parse_double(require_value(i, argc, argv), arg);
        else if(arg == "--check-th") opt.check_th = parse_double(require_value(i, argc, argv), arg);
        else if(arg == "--no-semantic") opt.semantic = false;
        else if(arg == "--async-update") opt.async_update = true;
        else if(arg == "--registration-backend")
        {
            string backend = require_value(i, argc, argv);
            if(backend == "ransac" || backend == "RANSAC")
            {
                opt.registration_backend = iBoW3D::RegistrationBackend::RANSAC;
            }
            else if(backend == "fgr" || backend == "FGR")
            {
                opt.registration_backend = iBoW3D::RegistrationBackend::FGR;
            }
            else
            {
                throw runtime_error("Unsupported registration backend: " + backend);
            }
        }
        else
        {
            throw runtime_error("Unknown argument: " + arg);
        }
    }
    return opt;
}

void complete_defaults(Options& opt)
{
    if(opt.dataset.empty() || opt.sequence.empty())
    {
        throw runtime_error("--dataset and --sequence are required");
    }

    opt.project_root = trim_trailing_slash(opt.project_root);

    if(opt.scan_dir.empty())
    {
        opt.scan_dir = join_path(opt.project_root, "res_data/" + opt.dataset + "/" + opt.sequence + "/D3F_allpoints");
    }
    if(opt.pose_file.empty())
    {
        if(opt.dataset == "KITTI360")
        {
            opt.pose_file = join_path(opt.project_root, "data/KITTI360/2013_05_28_drive_00" + opt.sequence + "_sync/GTposes.csv");
        }
        else
        {
            opt.pose_file = join_path(opt.project_root, "data/" + opt.dataset + "/" + opt.sequence + "/GTposes.csv");
        }
    }
    if(opt.loop_file.empty())
    {
        if(opt.dataset == "KITTI360")
        {
            opt.loop_file = join_path(opt.project_root, "data/KITTI360/2013_05_28_drive_00" + opt.sequence + "_sync/loop_lst.csv");
        }
        else
        {
            opt.loop_file = join_path(opt.project_root, "data/" + opt.dataset + "/" + opt.sequence + "/loop_lst.csv");
        }
    }
    if(opt.key_feature_dir.empty())
    {
        opt.key_feature_dir = join_path(opt.project_root, "descriptor_txt/D3F/" + opt.dataset + "/" + opt.sequence);
    }
    if(opt.all_feature_dir.empty())
    {
        opt.all_feature_dir = join_path(opt.project_root, "feature_txt/D3F/" + opt.dataset + "/" + opt.sequence);
    }
    if(opt.label_dir.empty())
    {
        opt.label_dir = join_path(opt.project_root, "res_data/" + opt.dataset + "/" + opt.sequence + "/D3F_keypoints_label_reset_SF");
    }
    if(opt.result_dir.empty())
    {
        opt.result_dir = join_path(opt.project_root, "results/" + opt.dataset + "_" + opt.sequence);
    }
    if(opt.save_time && opt.time_dir.empty())
    {
        opt.time_dir = join_path(opt.result_dir, "time_info");
    }

    opt.scan_dir = ensure_trailing_slash(opt.scan_dir);
    opt.key_feature_dir = ensure_trailing_slash(opt.key_feature_dir);
    opt.all_feature_dir = ensure_trailing_slash(opt.all_feature_dir);
    opt.label_dir = ensure_trailing_slash(opt.label_dir);
    opt.result_dir = ensure_trailing_slash(opt.result_dir);
    opt.time_dir = ensure_trailing_slash(opt.time_dir);

    if(opt.fit_th < 0.0)
    {
        opt.fit_th = opt.dataset == "KITTI360" ? 0.90 : 0.94;
    }
    if(opt.score_th < 0.0)
    {
        opt.score_th = opt.dataset == "KITTI360" ? 1.6 : 1.4;
    }
    if(opt.fit_th_2 < 0.0)
    {
        opt.fit_th_2 = opt.fit_th;
    }
    if(opt.score_th_2 < 0.0)
    {
        opt.score_th_2 = opt.score_th;
    }
    if(opt.ransac_n <= 0)
    {
        opt.ransac_n = opt.dataset == "CU" ? 3 : 4;
    }

    if(opt.init_pcd_num < 2)
    {
        throw runtime_error("--init-pcd-num must be at least 2 because dictionary initialization needs two or more frames");
    }
    if(opt.update_num <= 0 || opt.keypoint_num <= 0 || opt.feature_dim <= 0)
    {
        throw runtime_error("--update-num, --keypoint-num, and --feature-dim must be positive");
    }
    if(opt.init_words_num <= 0 || opt.words_num_add <= 0 || opt.search_num <= 0)
    {
        throw runtime_error("--init-words-num, --words-num-add, and --search-num must be positive");
    }
    if(opt.near_num <= 0)
    {
        throw runtime_error("--near-num must be positive");
    }
    if(opt.gap_num < 0)
    {
        throw runtime_error("--gap-num must be non-negative");
    }
    if(opt.max_iter <= 0 || opt.ransac_n <= 0)
    {
        throw runtime_error("--max-iter and --ransac-n must be positive");
    }
    if(opt.semantic_num <= 0)
    {
        throw runtime_error("--semantic-num must be positive");
    }
    if(!isfinite(opt.lambda_word) || opt.lambda_word < 0.0)
    {
        throw runtime_error("--lambda-word must be a finite non-negative number");
    }
    if(!isfinite(opt.fit_th) || !isfinite(opt.score_th) ||
       !isfinite(opt.fit_th_2) || !isfinite(opt.score_th_2) ||
       !isfinite(opt.check_th))
    {
        throw runtime_error("registration and range-check thresholds must be finite");
    }
    if(opt.check_th <= 0.0)
    {
        throw runtime_error("--check-th must be positive");
    }
}

void validate_inputs(const Options& opt)
{
    if(!dir_exists(opt.scan_dir))
    {
        throw runtime_error("Point cloud directory does not exist: " + opt.scan_dir);
    }
    if(!file_exists(opt.pose_file))
    {
        throw runtime_error("Pose file does not exist: " + opt.pose_file);
    }
    if(!file_exists(opt.loop_file))
    {
        throw runtime_error("Loop file does not exist: " + opt.loop_file);
    }
    if(!dir_exists(opt.key_feature_dir))
    {
        throw runtime_error("Key-feature directory does not exist: " + opt.key_feature_dir);
    }
    if(!dir_exists(opt.all_feature_dir))
    {
        throw runtime_error("All-feature directory does not exist: " + opt.all_feature_dir);
    }
    if(opt.semantic && !dir_exists(opt.label_dir))
    {
        throw runtime_error("Semantic label directory does not exist: " + opt.label_dir);
    }
}

string scan_path(const Options& opt, int frame_id)
{
    return opt.scan_dir + to_string(frame_id) + ".ply";
}

shared_ptr<geometry::PointCloud> load_point_cloud_or_throw(const string& path)
{
    if(!file_exists(path))
    {
        throw runtime_error("Point cloud file does not exist: " + path);
    }

    auto pcd = open3d::io::CreatePointCloudFromFile(path);
    if(!pcd || pcd->IsEmpty())
    {
        throw runtime_error("Failed to read a non-empty point cloud: " + path);
    }
    return pcd;
}

void write_scalar_file(const string& path, const vector<double>& values)
{
    ofstream out(path);
    if(!out.is_open())
    {
        throw runtime_error("Failed to write: " + path);
    }
    for(const auto value : values)
    {
        out << value << "\n";
    }
}

void write_pair_file(const string& path, const vector<pair<int, double>>& values)
{
    ofstream out(path);
    if(!out.is_open())
    {
        throw runtime_error("Failed to write: " + path);
    }
    for(const auto& value : values)
    {
        out << value.first << " " << value.second << "\n";
    }
}

void write_time_info(const string& time_dir)
{
    if(time_dir.empty())
    {
        return;
    }
    if(!create_directories_if_needed(time_dir))
    {
        throw runtime_error("Failed to create timing directory: " + time_dir);
    }

    write_scalar_file(time_dir + "processTime.txt", processTime);
    write_scalar_file(time_dir + "CandidateSelectionTime.txt", CandidateSelectionTime);
    write_scalar_file(time_dir + "GeoVerificationTime.txt", GeoVerificationTime);
    write_scalar_file(time_dir + "RangeCheckTime.txt", RangeCheckTime);
    write_scalar_file(time_dir + "RegistrationTime.txt", RegistrationTime);
    write_scalar_file(time_dir + "NoLoopRegistrationTime.txt", NoLoopRegistrationTime);
    write_scalar_file(time_dir + "WithLoopRegistrationTime.txt", WithLoopRegistrationTime);
    write_pair_file(time_dir + "UpdateBoWTime.txt", UpdateBoWTime);
    write_pair_file(time_dir + "UpdateBoWClusterTime.txt", UpdateBoWClusterTime);
    write_pair_file(time_dir + "UpdateBoWUpdateDatabaseTime.txt", UpdateBoWUpdateDatabaseTime);
    write_pair_file(time_dir + "UpdateBoWForegroundTime.txt", UpdateBoWForegroundTime);
}

void print_run_config(const Options& opt)
{
    cout << "------------ Sem-iBoW3D configuration ------------\n";
    cout << "dataset: " << opt.dataset << "\n";
    cout << "sequence: " << opt.sequence << "\n";
    cout << "scan_dir: " << opt.scan_dir << "\n";
    cout << "pose_file: " << opt.pose_file << "\n";
    cout << "loop_file: " << opt.loop_file << "\n";
    cout << "key_feature_dir: " << opt.key_feature_dir << "\n";
    cout << "all_feature_dir: " << opt.all_feature_dir << "\n";
    cout << "label_dir: " << opt.label_dir << "\n";
    cout << "result_dir: " << opt.result_dir << "\n";
    cout << "time_dir: " << opt.time_dir << "\n";
    cout << "semantic: " << (opt.semantic ? 1 : 0) << "\n";
    cout << "async_update: " << (opt.async_update ? 1 : 0) << "\n";
    cout << "fit_th: " << opt.fit_th << "\n";
    cout << "score_th: " << opt.score_th << "\n";
    cout << "fit_th_2: " << opt.fit_th_2 << "\n";
    cout << "score_th_2: " << opt.score_th_2 << "\n";
    cout << "search_num: " << opt.search_num << "\n";
    cout << "max_iter: " << opt.max_iter << "\n";
    cout << "check_th: " << opt.check_th << "\n";
    cout << "ransac_n: " << opt.ransac_n << "\n";
    cout << "init_pcd_num: " << opt.init_pcd_num << "\n";
    cout << "update_num: " << opt.update_num << "\n";
    cout << "init_words_num: " << opt.init_words_num << "\n";
    cout << "words_num_add: " << opt.words_num_add << "\n";
    cout << "lambda_word: " << opt.lambda_word << "\n";
}

void write_results(const Options& opt,
                   double retrieve_time_sum,
                   int retrieve_time_count,
                   int tp1,
                   int tp2,
                   int fp,
                   int fn,
                   const vector<pair<int, int>>& loop_list,
                   const vector<int>& fn_list)
{
    if(!create_directories_if_needed(opt.result_dir))
    {
        throw runtime_error("Failed to create result directory: " + opt.result_dir);
    }

    int tp = tp1 + tp2;
    double precision = (tp + fp) > 0 ? static_cast<double>(tp) / static_cast<double>(tp + fp) : 0.0;
    double recall = (tp + fn) > 0 ? static_cast<double>(tp) / static_cast<double>(tp + fn) : 0.0;
    double f1 = (precision + recall) > 0.0 ? 2.0 * precision * recall / (precision + recall) : 0.0;
    double retrieve_time_avg = retrieve_time_count > 0 ? retrieve_time_sum / retrieve_time_count : 0.0;

    ofstream result_file(opt.result_dir + "results.txt");
    if(!result_file.is_open())
    {
        throw runtime_error("Failed to write results.txt in " + opt.result_dir);
    }

    result_file << "dataset: " << opt.dataset << "\n";
    result_file << "seq: " << opt.sequence << "\n";
    result_file << "is_semantic: " << opt.semantic << "\n";
    result_file << "key_label_path: " << opt.label_dir << "\n";
    result_file << "fit_th: " << opt.fit_th << "\n";
    result_file << "score_th: " << opt.score_th << "\n";
    result_file << "fit_th_2: " << opt.fit_th_2 << "\n";
    result_file << "score_th_2: " << opt.score_th_2 << "\n";
    result_file << "search_num: " << opt.search_num << "\n";
    result_file << "max_iter: " << opt.max_iter << "\n";
    result_file << "check_th: " << opt.check_th << "\n";
    result_file << "init_pcd_num: " << opt.init_pcd_num << "\n";
    result_file << "update_num: " << opt.update_num << "\n";
    result_file << "lambda_word: " << opt.lambda_word << "\n";
    result_file << "init_words_num: " << opt.init_words_num << "\n";
    result_file << "words_num_add: " << opt.words_num_add << "\n";
    result_file << "near_num: " << opt.near_num << "\n";
    result_file << "gap_num: " << opt.gap_num << "\n";
    result_file << "ransac_n: " << opt.ransac_n << "\n";
    result_file << "retrieve_time_sum: " << retrieve_time_sum << "\n";
    result_file << "retrieve_time_cnt: " << retrieve_time_count << "\n";
    result_file << "retrieve_time_avg: " << retrieve_time_avg << "\n";
    result_file << "TP1: " << tp1 << "\n";
    result_file << "TP2: " << tp2 << "\n";
    result_file << "TP: " << tp << "\n";
    result_file << "FP: " << fp << "\n";
    result_file << "FN: " << fn << "\n";
    result_file << "P: " << precision << "\n";
    result_file << "R: " << recall << "\n";
    result_file << "F1: " << f1 << "\n";

    ofstream loop_file(opt.result_dir + "looplist.txt");
    if(!loop_file.is_open())
    {
        throw runtime_error("Failed to write looplist.txt in " + opt.result_dir);
    }
    for(const auto& loop_pair : loop_list)
    {
        loop_file << loop_pair.first << ", " << loop_pair.second << "\n";
    }

    ofstream fn_file(opt.result_dir + "false_negative_frames.txt");
    if(!fn_file.is_open())
    {
        throw runtime_error("Failed to write false_negative_frames.txt in " + opt.result_dir);
    }
    for(const auto frame_id : fn_list)
    {
        fn_file << frame_id << "\n";
    }

    cout << "------------ results ------------\n";
    cout << "TP1: " << tp1 << "\n";
    cout << "TP2: " << tp2 << "\n";
    cout << "TP: " << tp << "\n";
    cout << "FP: " << fp << "\n";
    cout << "FN: " << fn << "\n";
    cout << "P: " << precision << "\n";
    cout << "R: " << recall << "\n";
    cout << "F1: " << f1 << "\n";
    cout << "results saved to: " << opt.result_dir << "\n";
}
}

int main(int argc, char** argv)
{
    try
    {
        Options opt = parse_args(argc, argv);
        complete_defaults(opt);
        validate_inputs(opt);
        print_run_config(opt);

        vector<vector<float>> gt_pose = iBoW3D::read_ground_truth_pose(opt.pose_file);
        vector<int> frame_loop = iBoW3D::read_loop_ID(opt.loop_file);
        if(gt_pose.empty())
        {
            throw runtime_error("No poses were loaded from: " + opt.pose_file);
        }

        int num_data = static_cast<int>(gt_pose.size());
        if(opt.max_frames > 0 && opt.max_frames < num_data)
        {
            num_data = opt.max_frames;
            cout << "Limit frames by --max-frames: " << num_data << "\n";
        }
        cout << "The number of sequence frames is: " << num_data << "\n";
        cout << "The number of GT loop frame ids is: " << frame_loop.size() << "\n";

        int frame_id = 0;
        timeval t1{}, t2{}, t3{}, t4{};
        gettimeofday(&t1, nullptr);
        auto current_pcd = load_point_cloud_or_throw(scan_path(opt, frame_id));
        gettimeofday(&t2, nullptr);
        cout << "initial scan loading time: " << seconds_between(t1, t2) << "\n";

        unique_ptr<iBoW3D::DataBase> database(new iBoW3D::DataBase());
        unique_ptr<iBoW3D::iBoW3D> recognizer(new iBoW3D::iBoW3D(
            current_pcd,
            frame_id,
            opt.init_words_num,
            opt.words_num_add,
            opt.keypoint_num,
            opt.feature_dim,
            opt.key_feature_dir,
            opt.all_feature_dir,
            database.get(),
            opt.lambda_word,
            opt.near_num,
            opt.search_num,
            opt.score_th,
            opt.fit_th,
            opt.check_th,
            opt.score_th_2,
            opt.fit_th_2,
            true,
            opt.gap_num,
            opt.max_iter,
            opt.ransac_n,
            false,
            opt.semantic,
            opt.semantic_num,
            opt.label_dir));

        recognizer->set_async_update(opt.async_update);
        recognizer->set_registration_backend(opt.registration_backend);
        recognizer->update_database();

        int tp1 = 0;
        int tp2 = 0;
        int fp = 0;
        int fn = 0;
        int added_num = 0;
        double retrieve_time_sum = 0.0;
        int retrieve_time_count = 0;
        vector<pair<int, int>> loop_list;
        vector<pair<int, int>> fp_list;
        vector<int> fn_list;

        for(frame_id = 1; frame_id < num_data; frame_id++)
        {
            gettimeofday(&t1, nullptr);
            string current_scan_path = scan_path(opt, frame_id);
            cout << current_scan_path << "\n";
            current_pcd = load_point_cloud_or_throw(current_scan_path);
            recognizer->update_current_frame(current_pcd, frame_id);
            gettimeofday(&t2, nullptr);
            cout << "scan + feature loading time: " << seconds_between(t1, t2) << "\n";

            if(frame_id < opt.init_pcd_num - 1)
            {
                recognizer->update_database();
                continue;
            }
            if(frame_id == opt.init_pcd_num - 1)
            {
                recognizer->update_database();
                recognizer->get_dictionary_and_histogram();
                continue;
            }

            if(frame_id >= static_cast<int>(gt_pose.size()))
            {
                throw runtime_error("Pose index out of range at frame " + to_string(frame_id));
            }

            vector<float> current_location = gt_pose[frame_id];
            recognizer->get_current_histogram();
            cout << "start retrieve\n";

            gettimeofday(&t3, nullptr);
            int loop_id = recognizer->retrieve();
            gettimeofday(&t4, nullptr);

            double retrieve_time = seconds_between(t3, t4);
            cout << "process time: " << retrieve_time << "\n";
            retrieve_time_sum += retrieve_time;
            retrieve_time_count += 1;
            processTime.push_back(retrieve_time);

            if(loop_id == -1)
            {
                cout << "No Loop of frame: " << frame_id << "\n";
                if(find(frame_loop.begin(), frame_loop.end(), frame_id) != frame_loop.end())
                {
                    fn++;
                    fn_list.push_back(frame_id);
                    cout << "False Negative!\n";
                }

                added_num++;
                recognizer->update_database();
                if(added_num == opt.update_num)
                {
                    recognizer->update_dictionary_histograms();
                    added_num = 0;
                }
            }
            else
            {
                cout << "************* Loop! ************\n";
                cout << frame_id << " Loop with frame: " << loop_id << "\n";

                recognizer->update_only_DBIdx();
                Eigen::Matrix4d_u trans = recognizer->get_trans();

                if(loop_id < 0 || loop_id >= static_cast<int>(gt_pose.size()))
                {
                    throw runtime_error("Predicted loop id is outside pose range: " + to_string(loop_id));
                }
                vector<float> loop_location = gt_pose[loop_id];
                double gt_dist = hypot(current_location[0] - loop_location[0],
                                       current_location[1] - loop_location[1]);
                double t_dist = sqrt(pow(trans(0, 3), 2) + pow(trans(1, 3), 2) + pow(trans(2, 3), 2));
                double delta_dist = abs(gt_dist - t_dist);
                if(gt_dist <= 4.0)
                {
                    tp1++;
                    loop_list.push_back(make_pair(frame_id, loop_id));
                }
                else if(delta_dist <= 4.0)
                {
                    tp2++;
                    loop_list.push_back(make_pair(frame_id, loop_id));
                }
                else
                {
                    fp++;
                    fp_list.push_back(make_pair(frame_id, loop_id));
                }
            }
        }

        recognizer->wait_for_async_update();
        write_results(opt, retrieve_time_sum, retrieve_time_count, tp1, tp2, fp, fn, loop_list, fn_list);
        if(opt.save_time)
        {
            write_time_info(opt.time_dir);
        }
    }
    catch(const exception& exc)
    {
        cerr << "Sem-iBoW3D error: " << exc.what() << "\n";
        return 1;
    }

    return 0;
}
