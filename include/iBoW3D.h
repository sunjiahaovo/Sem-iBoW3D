#pragma once

#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <execution>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
// #include <Eigen/Dense>
// #include <Eigen/Core>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <open3d/Open3D.h>

#include <math.h>

#include "Data_IO.h"
#include "FeatureContainer.h"
#include "DataBase.h"
#include "LabelContainer.h"

#include "TimeSave.h"

#include <stdio.h>
#include <sys/time.h>

using namespace open3d;
using namespace std;

namespace iBoW3D
{
    struct result_struct
    {
        double fit = 0.0;
        double score = 100.0;
        Eigen::Matrix4d_u trans = Eigen::Matrix4d_u::Identity();
        int idx = -1;
    };

    struct cand_info
    {
        std::shared_ptr<geometry::PointCloud> pCand;
        std::shared_ptr<pipelines::registration::Feature> cand_features;
        int idx;
    };

    struct reg_info
    {
        std::shared_ptr<geometry::PointCloud> current_pcd;
        std::shared_ptr<pipelines::registration::Feature> current_features;

        std::shared_ptr<geometry::PointCloud> pCand;
        std::shared_ptr<pipelines::registration::Feature> cand_features;
        int idx;
    };

    enum class RegistrationBackend
    {
        RANSAC,
        FGR
    };


    class iBoW3D
    {
        public:
            iBoW3D(std::shared_ptr<geometry::PointCloud> current_pcd_,
                   int frameID_, int init_words_num_, int words_num_add_,
                   const int keypoint_num_, const int feature_dim_,
                   const std::string key_feature_path_, const std::string all_feature_path_,
                   DataBase *pDB_,
                   double lambda_word_, int near_num_, int search_num_,
                   double score_th_, double fit_th_, double chech_th_, double score_th_2_, double fit_th_2_,
                   bool prior_fit_, int gap_num_,
                   int max_iter_, int ransac_n_, bool remove_outliers_,
                   bool is_semantic_, int semantic_num_, const std::string key_label_path_, double lambda_label_);

            ~iBoW3D()
            {
                wait_for_async_update();
                delete pFC;
                delete pLC;
            };

            // update current pcd
            void update_current_frame(std::shared_ptr<geometry::PointCloud> current_pcd_, int frameID_);

            static void validate_label_value(int label, int semantic_num, const std::string& context);
            static double mean_candidate_distance(const std::vector<std::pair<double, int>>& distance_list,
                                                  int center_frame_id,
                                                  int near_num,
                                                  std::vector<int>* island = nullptr);

            // // update last loop ID
            // void update_lastLoopID(int newID){lastLoopID = newID;}

            // if is_semantic is true, then the key_label_path needs to be set
            void set_key_label_path(const std::string key_label_path_);

            // create BoW dictionary; get and save histograms of each seen scans
            void get_dictionary_and_histogram();

            // update BoW dictionary
            void update_dictionary_histograms();
            void set_async_update(bool enabled){ async_update_enabled = enabled; }
            bool is_async_update_enabled() const { return async_update_enabled; }
            void set_registration_backend(RegistrationBackend backend){ registration_backend = backend; }
            void apply_async_update_if_ready();
            void wait_for_async_update();

            // update database (can also initialize)
            void update_database();
            void update_only_DBIdx();
            // void update_database_wo_key_feat();

            // get current histogram
            void get_current_histogram();

            // retrieve the loop ID
            int retrieve();
            vector<vector<int>> cand_selection();
            // int geo_verification_icp(vector<int> cand_list);
            int geo_verification_ransac(vector<vector<int>> cand_list);
            pipelines::registration::RegistrationResult run_registration(
                                    const geometry::PointCloud& source,
                                    const geometry::PointCloud& target,
                                    const pipelines::registration::Feature& source_feature,
                                    const pipelines::registration::Feature& target_feature);
            void pcd_ransac(pair<int, int> cand_pair, vector<result_struct> & res_list,
                                    std::shared_ptr<geometry::PointCloud> current_pcd,
                                    std::shared_ptr<pipelines::registration::Feature> current_features);
            // void pcd_ransac(pair<int, reg_info> cand_info_pair, vector<result_struct> & res_list);
            void island_ransac(pair<int, vector<int>> island_pair, vector<result_struct> & res_list,
                               std::shared_ptr<geometry::PointCloud> current_pcd,
                               std::shared_ptr<pipelines::registration::Feature> current_features);
            vector<vector<int>> range_check(vector<vector<int>> cand_list);

            // get transformation
            Eigen::Matrix4d_u get_trans(){ return trans; }



        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            int frameID;
            int lastLoopID;

            int LoopID;

        private:
            // current frame info
            std::shared_ptr<geometry::PointCloud> current_pcd;
            int pcd_num;

            const std::string key_feature_path;
            const std::string all_feature_path;

            FeatureContainer *pFC;
            Eigen::VectorXd current_hist;

            // database
            DataBase *pDB;
            vector<Eigen::VectorXd> hist_DB;
            vector<Eigen::VectorXd> label_hist_DB;

            // BoW dictionary
            int words_num; // the number of words in the dictionary
            int words_num_add; // the added number of words in the dictionary
            int new_words_num;

            cv::Mat labels; // save class label of each feature
            cv::Mat centers; // save the data of each center of cluster
            // vector<cv::Mat> labels_list;
            vector<cv::Mat> centers_list;
            vector<int> cluster_num_list;
            vector<int> start_idx_list;


            // Mat centers(words_num,1,points.type()
            Eigen::ArrayXd idf;


            // parameters about features
            int keypoint_num;
            int feature_dim;

            // hyperparameters
            double lambda_word;
            double lambda_label;

            int near_num;
            int search_num;
            double score_th;
            double fit_th;
            double check_th;
            double score_th_2;
            double fit_th_2;
            bool prior_fit;
            int gap_num;
            int max_iter;
            int ransac_n;
            RegistrationBackend registration_backend = RegistrationBackend::RANSAC;

            int distance_length;

            // remove outliers
            bool remove_outliers;

            // registration result
            // vector<result_struct> res_list;
            Eigen::Matrix4d_u trans;

            // semantic or not
            bool is_semantic;
            std::string key_label_path;

            LabelContainer *pLC;
            int semantic_num;
            Eigen::VectorXd current_label_hist;
            Eigen::ArrayXd label_idf;

            struct BowBuildResult
            {
                int words_num = 0;
                int new_words_num = 0;
                int db_len = 0;
                cv::Mat labels;
                cv::Mat centers;
                vector<cv::Mat> centers_list;
                vector<int> cluster_num_list;
                vector<int> start_idx_list;
                Eigen::ArrayXd idf;
                vector<Eigen::VectorXd> hist_DB;
                double cluster_time_s = 0.0;
                double database_time_s = 0.0;
            };

            BowBuildResult build_dictionary_from_snapshot(int target_words_num,
                                                          const vector<cv::Mat>& key_feat_DB,
                                                          const vector<cv::Mat>& key_label_DB) const;
            void apply_dictionary_build_result(const BowBuildResult& result);
            void snapshot_database(vector<cv::Mat>& key_feat_DB,
                                   vector<cv::Mat>& key_label_DB) const;
            void launch_async_update_dictionary_histograms();
            void start_async_update_dictionary_histograms();

            bool async_update_enabled = false;
            std::atomic<bool> async_update_running{false};
            std::atomic<bool> async_update_pending{false};
            std::thread async_update_thread;
            std::mutex async_update_result_mutex;
            std::unique_ptr<BowBuildResult> async_update_result;
    };
}
