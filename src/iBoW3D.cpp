#include "iBoW3D.h"

#include <chrono>
#include <limits>

using namespace cv;
using namespace std;

namespace iBoW3D
{
    iBoW3D::iBoW3D(std::shared_ptr<geometry::PointCloud> current_pcd_,
                   int frameID_, int init_words_num_, int words_num_add_,
                   int keypoint_num_, int feature_dim_,
                   const std::string key_feature_path_, const std::string all_feature_path_,
                   DataBase *pDB_,
                   double lambda_word_, int near_num_, int search_num_,
                   double score_th_, double fit_th_, double check_th_, double score_th_2_, double fit_th_2_,
                   bool prior_fit_, int gap_num_,
                   int max_iter_, int ransac_n_, bool remove_outliers_,
                   bool is_semantic_, int semantic_num_, const std::string key_label_path_):
    frameID(frameID_), lastLoopID(-100), LoopID(-1),
    current_pcd(current_pcd_), pcd_num(0),
    key_feature_path(key_feature_path_), all_feature_path(all_feature_path_),
    pFC(nullptr),
    pDB(pDB_),
    words_num(init_words_num_), words_num_add(words_num_add_),
    new_words_num(0),
    keypoint_num(keypoint_num_), feature_dim(feature_dim_),
    lambda_word(lambda_word_),
    near_num(near_num_), search_num(search_num_),
    score_th(score_th_), fit_th(fit_th_), check_th(check_th_), score_th_2(score_th_2_), fit_th_2(fit_th_2_),
    prior_fit(prior_fit_), gap_num(gap_num_),
    max_iter(max_iter_), ransac_n(ransac_n_), distance_length(100), remove_outliers(remove_outliers_),
    trans(Eigen::Matrix4d_u::Identity()),
    is_semantic(is_semantic_), key_label_path(key_label_path_), pLC(nullptr), semantic_num(semantic_num_)
    {
        // lastLoopID = -100;
        // LoopID = -1;

        // current frame info
        pcd_num = current_pcd->points_.size();
        auto initialFC = std::unique_ptr<FeatureContainer>(
            new FeatureContainer(keypoint_num, feature_dim, pcd_num, frameID, key_feature_path, all_feature_path));

        if(is_semantic)
        {
            auto initialLC = std::unique_ptr<LabelContainer>(
                new LabelContainer(keypoint_num, frameID, key_label_path, semantic_num));
            pLC = initialLC.release();
        }
        pFC = initialFC.release();

        /* current_hist = Mat::zeros(Size(1,words_num), CV_16S); */
        current_hist.resize(words_num, 1);
        current_hist.setZero();

        idf = Eigen::ArrayXd::Zero(words_num);
        // res_list.clear();

    }


    void iBoW3D::update_current_frame(std::shared_ptr<geometry::PointCloud> current_pcd_, int frameID_)
    {
        int next_pcd_num = current_pcd_->points_.size();
        auto nextFC = std::unique_ptr<FeatureContainer>(
            new FeatureContainer(keypoint_num, feature_dim, next_pcd_num, frameID_, key_feature_path, all_feature_path));
        std::unique_ptr<LabelContainer> nextLC;

        if(is_semantic)
        {
            nextLC.reset(new LabelContainer(keypoint_num, frameID_, key_label_path, semantic_num));
        }

        delete pFC;
        delete pLC;
        current_pcd = current_pcd_;
        frameID = frameID_;
        pcd_num = next_pcd_num;
        pFC = nextFC.release();
        pLC = nextLC.release();
    }

    void iBoW3D::validate_label_value(int label, int semantic_num, const std::string& context)
    {
        if(label < -1 || label >= semantic_num)
        {
            throw runtime_error("Semantic label out of range at " + context + ": " + to_string(label) +
                                " allowed range is -1 or [0, " + to_string(semantic_num - 1) + "]");
        }
    }

    double iBoW3D::mean_candidate_distance(const vector<pair<double, int>>& distance_list,
                                           int center_frame_id,
                                           int near_num,
                                           vector<int>* island)
    {
        if(near_num <= 0)
        {
            throw runtime_error("near_num must be positive");
        }

        double dist_sum = 0.0;
        int idx_num = 0;
        if(island)
        {
            island->clear();
        }

        for(const auto& candidate : distance_list)
        {
            if(abs(candidate.second - center_frame_id) < near_num)
            {
                if(island)
                {
                    island->push_back(candidate.second);
                }
                dist_sum += candidate.first;
                idx_num++;
            }
        }

        if(idx_num == 0)
        {
            throw runtime_error("No candidates found inside near_num window for frame " + to_string(center_frame_id));
        }
        return dist_sum / static_cast<double>(idx_num);
    }

    void iBoW3D::snapshot_database(vector<cv::Mat>& key_feat_DB,
                                   vector<cv::Mat>& key_label_DB) const
    {
        key_feat_DB = pDB->get_key_feat();
        key_label_DB = pDB->get_key_label();

        for(auto& key_feat : key_feat_DB)
        {
            key_feat = key_feat.clone();
        }
        for(auto& key_label : key_label_DB)
        {
            key_label = key_label.clone();
        }
    }

    iBoW3D::BowBuildResult iBoW3D::build_dictionary_from_snapshot(int target_words_num,
                                                                  const vector<cv::Mat>& key_feat_DB,
                                                                  const vector<cv::Mat>& key_label_DB) const
    {
        BowBuildResult result;
        result.words_num = target_words_num;
        result.db_len = static_cast<int>(key_feat_DB.size());

        if(not is_semantic)
        {
            struct timeval t1, t2, t3, t4;
            gettimeofday(&t1, NULL);

            // get dictionary
            int DB_len = result.db_len;
            int key_feat_total_num = keypoint_num * DB_len;
            Mat keyFeatures(key_feat_total_num, feature_dim, CV_32F, Scalar(0));
            Mat temp;

            Mat keyLabels(key_feat_total_num, 1, CV_32S, Scalar(-1));
            Mat temp2;
            for(int i = 0; i < DB_len; i++)
            {
                temp = keyFeatures(Rect(0, i*keypoint_num, feature_dim, keypoint_num));
                key_feat_DB[i].copyTo(temp);
            }

            // implement k-means with k-means++ seeds
            TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1);
            kmeans(keyFeatures, target_words_num, result.labels, criteria, 3, KMEANS_PP_CENTERS, result.centers);

            // calculate idf
            Eigen::ArrayXd idf_temp = Eigen::ArrayXd::Zero(target_words_num);
            result.idf = Eigen::ArrayXd::Zero(target_words_num); // set every element of idf list = 0

            for(int i = 0; i < key_feat_total_num; i++)
            {
                idf_temp(result.labels.at<int>(i))++;
                // cout << i << endl;
                // cout << labels.at<int>(i) << endl;
            }
            for(int i = 0; i < target_words_num; i++)
            {
                result.idf(i) = log(key_feat_total_num/(idf_temp(i)+1));
                // cout << i << endl;
                // cout << idf_temp(i) << endl;
                // cout << idf(i) << endl;
            }

            gettimeofday(&t2, NULL);
            result.cluster_time_s = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
            gettimeofday(&t3, NULL);

            // get histograms
            result.hist_DB.clear();
            for(int i = 0; i < DB_len; i++)
            {
                /* Mat hist_temp = Mat::zeros(Size(1,words_num), CV_16S);
                Mat hist_temp_normalized = Mat::zeros(Size(1,words_num), CV_16S); */
                Eigen::VectorXd hist_temp = Eigen::VectorXd::Zero(target_words_num);

                for(int j = 0; j < keypoint_num; j++)
                {
                    hist_temp(result.labels.at<int>(i*keypoint_num+j)) += 1.0;
                }

                /* normalize(hist_temp, hist_temp_normalized, 1.0, 0.0, NORM_L2);
                hist_DB.push_back(hist_temp_normalized); */

                // get tf: hist_temp is divided by keypoint_num which is equal the number of features of current scan
                hist_temp = hist_temp / keypoint_num;
                // then multiple idf
                hist_temp = (hist_temp.array()*result.idf).matrix();
                // l2 normalize
                hist_temp.normalize();
                // add to database
                result.hist_DB.push_back(hist_temp);
            }

            gettimeofday(&t4, NULL);
            result.database_time_s = (t4.tv_sec - t3.tv_sec) + (double)(t4.tv_usec - t3.tv_usec)/1000000.0;
        }

        // if is semantic, get label histograms
        else
        {
            struct timeval t1, t2, t3, t4;
            gettimeofday(&t1, NULL);

            // get dictionary
            vector<cv::Mat> keyFeatures_list;
            vector<cv::Mat> labels_list;

            vector<int> label_num_list(semantic_num, 0);

            vector<vector<cv::Mat>> temp_keyFeature_l_list;
            labels_list.clear();
            temp_keyFeature_l_list.clear();
            for(int i=0; i<semantic_num; i++) // initialize above lists
            {
                Mat temp_labels, temp_centers;
                vector<cv::Mat> temp_keyFeature_l;

                labels_list.push_back(temp_labels);
                result.centers_list.push_back(temp_centers);
                temp_keyFeature_l_list.push_back(temp_keyFeature_l);
            }

            int DB_len = result.db_len;

            Mat temp_F, temp_L;
            int key_feat_sum = 0;
            for(int i = 0; i < DB_len; i++)
            {
                temp_F = key_feat_DB[i];
                temp_L = key_label_DB[i];
                for(int j=0; j<keypoint_num; j++)
                {
                    int label_idx = temp_L.at<int>(j);
                    validate_label_value(label_idx, semantic_num, "database label frame-index " + to_string(i) +
                                                              " keypoint " + to_string(j));
                    if(label_idx == -1)
                    {
                        continue;
                    }
                    temp_keyFeature_l_list[label_idx].push_back(temp_F.row(j));
                    label_num_list[label_idx]++;
                    key_feat_sum++;
                }
            }
            cout << "key_feat_sum 1: " << key_feat_sum << endl;
            if(key_feat_sum == 0)
            {
                throw runtime_error("No valid semantic keypoint labels were found while building the dictionary");
            }

            // obtain each element of keyFeatures_list
            for(int i=0; i<semantic_num; i++)
            {
                Mat keyFeatures;
                cv::vconcat(temp_keyFeature_l_list[i], keyFeatures);
                keyFeatures.convertTo(keyFeatures, CV_32F);
                keyFeatures_list.push_back(keyFeatures);
            }

            // calculate cluster numbers of each label, the number cannot be zero
            result.cluster_num_list.clear();
            result.cluster_num_list.resize(semantic_num, 0);
            // int last_cluster_num = words_num;
            result.new_words_num = 0;
            for(int i=0; i<semantic_num; i++)
            {
                if(label_num_list[i] == 0) // no features corresponded to the current label
                {
                    result.cluster_num_list[i] = 0;
                }
                else
                {
                    double cluster_num_d = target_words_num * (double)label_num_list[i]/(double)key_feat_sum;
                    int cluster_num = round(cluster_num_d);
                    if(cluster_num == 0)
                    {
                        cluster_num = 1;
                    }
                    result.cluster_num_list[i] = cluster_num;
                    // last_cluster_num -= cluster_num;
                    result.new_words_num += cluster_num;
                }
            }
            // cluster_num_list[semantic_num-1] = last_cluster_num;
            cout << "words_num: " << target_words_num << endl;
            cout << "new_words_num: " << result.new_words_num << endl;


            // implement k-means with k-means++ seeds for each label
            TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1);
            for(int i=0; i<semantic_num; i++)
            {
                if(result.cluster_num_list[i] == 0)
                {
                    continue;
                }
                kmeans(keyFeatures_list[i], result.cluster_num_list[i], labels_list[i], criteria, 3, KMEANS_PP_CENTERS, result.centers_list[i]);
                // cout << i << endl;
                // cout << cluster_num_list[i] << endl;
                // cout << centers_list[i].size().width << endl;
                // cout << centers_list[i].size().height << endl;
            }


            // calculate idf
            result.start_idx_list.clear();
            result.start_idx_list.resize(semantic_num, 0);
            int start_idx = 0;
            for(int i=0; i<semantic_num; i++)
            {
                result.start_idx_list[i] = start_idx;
                start_idx += result.cluster_num_list[i];
            }

            Eigen::ArrayXd idf_temp = Eigen::ArrayXd::Zero(result.new_words_num);
            result.idf = Eigen::ArrayXd::Zero(result.new_words_num); // set every element of idf list = 0

            key_feat_sum = 0;
            for(int i=0; i<semantic_num; i++)
            {
                if(result.cluster_num_list[i] == 0)
                {
                    continue;
                }

                Mat temp_labels = labels_list[i];
                for(int j=0; j<label_num_list[i]; j++)
                {
                    idf_temp(result.start_idx_list[i] + temp_labels.at<int>(j))++;
                    key_feat_sum++;
                }
            }
            cout << "key_feat_sum 2: " << key_feat_sum << endl;

            for(int i = 0; i < result.new_words_num; i++)
            {
                result.idf(i) = log(key_feat_sum/(idf_temp(i)+1));
            }


            gettimeofday(&t2, NULL);
            result.cluster_time_s = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
            cout << "UpdateBoWCluster time: " << result.cluster_time_s << endl;


            gettimeofday(&t3, NULL);
            // get histograms
            vector<int> processed_feat_num(semantic_num, 0);

            result.hist_DB.clear();
            for(int i = 0; i < DB_len; i++)
            {
                Eigen::VectorXd hist_temp = Eigen::VectorXd::Zero(result.new_words_num);
                temp_F = key_feat_DB[i];
                temp_L = key_label_DB[i];

                int added_key_feat_cnt = 0;
                for(int j = 0; j < keypoint_num; j++)
                {
                    int label_idx = temp_L.at<int>(j);
                    validate_label_value(label_idx, semantic_num, "database histogram label frame-index " + to_string(i) +
                                                              " keypoint " + to_string(j));
                    if(label_idx == -1)
                    {
                        continue;
                    }

                    if(result.cluster_num_list[label_idx] == 0)
                    {
                        continue;
                    }

                    hist_temp(result.start_idx_list[label_idx] + labels_list[label_idx].at<int>(processed_feat_num[label_idx])) += 1.0;
                    added_key_feat_cnt++;
                    processed_feat_num[label_idx]++;
                }

                // get tf: hist_temp is divided by keypoint_num which is equal the number of features of current scan
                if(added_key_feat_cnt == 0)
                {
                    result.hist_DB.push_back(hist_temp);
                    continue;
                }
                hist_temp = hist_temp / added_key_feat_cnt;
                // then multiple idf
                hist_temp = (hist_temp.array()*result.idf).matrix();
                // l2 normalize
                hist_temp.normalize();
                // add to database
                result.hist_DB.push_back(hist_temp);
            }

            gettimeofday(&t4, NULL);
            result.database_time_s = (t4.tv_sec - t3.tv_sec) + (double)(t4.tv_usec - t3.tv_usec)/1000000.0;
            cout << "UpdateBoWUpdateDatabase time: " << result.database_time_s << endl;

            // for(int i=0; i<semantic_num; i++)
            // {
            //     cout << "label_num_list " << i << ": " << label_num_list[i] << endl;
            //     cout << "processed_feat_num " << i << ": " << processed_feat_num[i] << endl;
            //     cout << endl;
            //     cout << "start_idx_list " << i << ": " << start_idx_list[i] << endl;
            //     cout << "cluster_num_list " << i << ": " << cluster_num_list[i] << endl;

            // }
        }

        return result;
    }


    void iBoW3D::apply_dictionary_build_result(const BowBuildResult& result)
    {
        words_num = result.words_num;
        new_words_num = result.new_words_num;
        labels = result.labels;
        centers = result.centers;
        centers_list = result.centers_list;
        cluster_num_list = result.cluster_num_list;
        start_idx_list = result.start_idx_list;
        idf = result.idf;
        hist_DB = result.hist_DB;

        UpdateBoWClusterTime.push_back(make_pair(result.db_len, result.cluster_time_s));
        UpdateBoWUpdateDatabaseTime.push_back(make_pair(result.db_len, result.database_time_s));
        UpdateBoWTime.push_back(make_pair(result.db_len, result.cluster_time_s + result.database_time_s));
    }


    void iBoW3D::get_dictionary_and_histogram()
    {
        int target_words_num = words_num;
        vector<cv::Mat> key_feat_DB;
        vector<cv::Mat> key_label_DB;
        snapshot_database(key_feat_DB, key_label_DB);
        BowBuildResult result = build_dictionary_from_snapshot(target_words_num, key_feat_DB, key_label_DB);
        apply_dictionary_build_result(result);
    }


    void iBoW3D::update_dictionary_histograms()
    {
        struct timeval t1, t2;
        gettimeofday(&t1, NULL);

        if(async_update_enabled)
        {
            start_async_update_dictionary_histograms();
            gettimeofday(&t2, NULL);
            double foreground_time = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
            int db_len = pDB->get_DB_len();
            UpdateBoWForegroundTime.push_back(make_pair(db_len, foreground_time));
            return;
        }

        cout << "------------------ Update Dictionary -----------------" << endl;
        words_num += words_num_add;
        get_dictionary_and_histogram();
        gettimeofday(&t2, NULL);
        double foreground_time = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
        int db_len = pDB->get_DB_len();
        UpdateBoWForegroundTime.push_back(make_pair(db_len, foreground_time));
    }


    void iBoW3D::launch_async_update_dictionary_histograms()
    {
        cout << "------------------ Launch Async Update Dictionary -----------------" << endl;
        int target_words_num = words_num + words_num_add;
        vector<cv::Mat> key_feat_DB;
        vector<cv::Mat> key_label_DB;
        snapshot_database(key_feat_DB, key_label_DB);

        if(async_update_thread.joinable())
        {
            async_update_thread.join();
        }

        async_update_running.store(true);
        async_update_pending.store(false);
        async_update_thread = std::thread(
            [this, target_words_num, key_feat_DB, key_label_DB]()
            {
                BowBuildResult result = build_dictionary_from_snapshot(target_words_num, key_feat_DB, key_label_DB);
                {
                    std::lock_guard<std::mutex> lock(async_update_result_mutex);
                    async_update_result.reset(new BowBuildResult(std::move(result)));
                }
                async_update_running.store(false);
            });
    }


    void iBoW3D::start_async_update_dictionary_histograms()
    {
        apply_async_update_if_ready();
        if(async_update_running.load())
        {
            async_update_pending.store(true);
            cout << "------------------ Async update still running; coalesce request -----------------" << endl;
            return;
        }

        launch_async_update_dictionary_histograms();
    }


    void iBoW3D::apply_async_update_if_ready()
    {
        if(async_update_enabled && !async_update_running.load() && async_update_thread.joinable())
        {
            async_update_thread.join();
        }

        std::unique_ptr<BowBuildResult> result;
        {
            std::lock_guard<std::mutex> lock(async_update_result_mutex);
            if(async_update_result)
            {
                result = std::move(async_update_result);
            }
        }

        if(result)
        {
            cout << "------------------ Apply Async Dictionary -----------------" << endl;
            apply_dictionary_build_result(*result);
            bool launch_next = async_update_pending.exchange(false);
            if(launch_next && async_update_enabled)
            {
                cout << "------------------ Launch coalesced Async Update Dictionary -----------------" << endl;
                launch_async_update_dictionary_histograms();
            }
        }
    }


    void iBoW3D::wait_for_async_update()
    {
        while(true)
        {
            if(async_update_thread.joinable())
            {
                async_update_thread.join();
                async_update_running.store(false);
            }

            std::unique_ptr<BowBuildResult> result;
            {
                std::lock_guard<std::mutex> lock(async_update_result_mutex);
                if(async_update_result)
                {
                    result = std::move(async_update_result);
                }
            }

            bool launch_next = false;
            if(result)
            {
                cout << "------------------ Apply Async Dictionary -----------------" << endl;
                apply_dictionary_build_result(*result);
                launch_next = async_update_pending.exchange(false);
            }
            else
            {
                async_update_pending.store(false);
            }

            if(launch_next && async_update_enabled)
            {
                cout << "------------------ Launch coalesced Async Update Dictionary -----------------" << endl;
                launch_async_update_dictionary_histograms();
                continue;
            }

            break;
        }
    }


    void iBoW3D::get_current_histogram()
    {
        apply_async_update_if_ready();

        if(not is_semantic)
        {
            /* current_hist = Scalar(0);
            Mat current_hist_no_norm = Mat::zeros(Size(1,words_num), CV_16S); */

            current_hist.resize(words_num,1);
            current_hist.setZero();

            Mat temp(1, feature_dim, CV_32F);

            vector<double> dist_list; // save distances
            int min_idx = 0;

            for(int i = 0; i < keypoint_num; i++)
            {
                dist_list.clear();
                temp = (pFC->getKeyFeature()).row(i);
                // identify the min index e.g. label
                for(int j = 0; j < words_num; j++)
                {
                    double dist = norm(temp-centers.row(j), NORM_L2);
                    dist_list.push_back(dist);
                }
                min_idx = distance(dist_list.begin(), min_element(dist_list.begin(), dist_list.end()));
                // corresponding number of word plus 1
                /* current_hist_no_norm.at<int>(0, min_idx) += 1; */
                current_hist(min_idx) += 1.0;
            }

            // get tf: current_hist is divided by keypoint_num which is equal the number of features of current scan
            current_hist = current_hist / keypoint_num;
            // then multiple idf
            current_hist = (current_hist.array()*idf).matrix();
            // l2 normalize
            current_hist.normalize();

            /* normalize(current_hist_no_norm, current_hist, 1.0, 0.0, NORM_L2); */
        }


        // get current label hist
        else
        {
            current_hist.resize(new_words_num, 1);
            current_hist.setZero();

            Mat temp(1, feature_dim, CV_32F);

            vector<double> dist_list; // save distances
            int min_idx = 0;

            int added_key_feat_cnt = 0;
            int temp_semantic_label;
            for(int i = 0; i < keypoint_num; i++)
            {
                dist_list.clear();
                temp = (pFC->getKeyFeature()).row(i);

                temp_semantic_label = (pLC->getKeyLabel()).at<int>(i,0);
                validate_label_value(temp_semantic_label, semantic_num, "current frame " + to_string(frameID) +
                                                              " keypoint " + to_string(i));
                if(temp_semantic_label == -1)
                {
                    continue;
                }

                // identify the min index e.g. label
                int temp_cluster_num = cluster_num_list[temp_semantic_label];
                if(temp_cluster_num == 0)
                {
                    continue;
                }

                if(temp_cluster_num == 1)
                {
                    min_idx = 0;
                }
                else
                {
                    // if(frameID == 2575) {cout << i << " temp_semantic_label: " << temp_semantic_label << " temp_cluster_num: " << temp_cluster_num << endl;}
                    // if(frameID == 2575) {cout << cluster_num_list[temp_semantic_label] << cluster_num_list[2] << endl;}

                    Mat temp_centers = centers_list[temp_semantic_label];
                    for(int j = 0; j < temp_cluster_num; j++)
                    {
                        // if(frameID == 2575) {cout << j << endl;
                        //                      cout << temp_centers.row(j).size().width << endl;
                        //                      cout << temp_centers.row(j).size().height << endl;
                        //                      cout << temp_centers.size().width << endl;
                        //                      cout << temp_centers.size().height << endl;}
                        double dist = norm(temp-temp_centers.row(j), NORM_L2);
                        // if(frameID == 2575) {cout << "dist: " << dist << endl;}
                        dist_list.push_back(dist);
                    }
                    min_idx = distance(dist_list.begin(), min_element(dist_list.begin(), dist_list.end()));
                }
                // corresponding number of word plus 1
                current_hist(start_idx_list[temp_semantic_label] + min_idx) += 1.0;
                added_key_feat_cnt++;
            }

            // get tf: current_hist is divided by keypoint_num which is equal the number of features of current scan
            if(added_key_feat_cnt == 0)
            {
                return;
            }
            current_hist = current_hist / added_key_feat_cnt;
            // then multiple idf
            current_hist = (current_hist.array()*idf).matrix();
            // l2 normalize
            current_hist.normalize();
        }

    }


    void iBoW3D::update_database()
    {
        apply_async_update_if_ready();

        pDB->add_pcloud(current_pcd);
        pDB->add_key_feat(pFC->getKeyFeature());
        pDB->add_all_feat(pFC->getAllFeature());
        pDB->add_frameIdx(frameID);
        pDB->add_DBIdx(frameID, 1); // DB_len has not added one
        pDB->add_DB_len();

        if(is_semantic)
        {
            pDB->add_key_label(pLC->getKeyLabel());
        }
    }

    void iBoW3D::update_only_DBIdx()
    {
        apply_async_update_if_ready();

        pDB->add_DBIdx(frameID, 0);
    }


    int iBoW3D::retrieve()
    {
        apply_async_update_if_ready();

        struct timeval t1, t2, t3, t4;
        double timeuse;

        gettimeofday(&t1, NULL);

        vector<vector<int>> cand_list = cand_selection();

        gettimeofday(&t2, NULL);
        timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
        cout << "cand_selection time: " << timeuse << endl;
        CandidateSelectionTime.push_back(timeuse);

        // cout << "Candidate List before Range Check: " << endl;
        // vector<vector<int>>::iterator iter;
        // for(iter=cand_list.begin(); iter!=cand_list.end(); iter++)
        // {
        //     for(int i=0; i<(int)((*iter).size()); i++)
        //     {
        //         cout << (*iter)[i] << " ";
        //     }
        //     cout << endl;
        // }
        // cout << endl;

        // if no loop, LoopID = -1
        if (cand_list.empty())
        {
            return -1;
        }
        else
        {
            vector<vector<int>> cand_list_2 = range_check(cand_list);

            gettimeofday(&t3, NULL);
            timeuse = (t3.tv_sec - t2.tv_sec) + (double)(t3.tv_usec - t2.tv_usec)/1000000.0;
            cout << "range_check time: " << timeuse << endl;
            RangeCheckTime.push_back(timeuse);

            if (cand_list_2.empty())
            {
                return -1;
            }
            else
            {
                cout << "Candidate List: " << endl;
                vector<vector<int>>::iterator iter;
                for(iter=cand_list_2.begin(); iter!=cand_list_2.end(); iter++)
                {
                    for(int i=0; i<(int)((*iter).size()); i++)
                    {
                        cout << (*iter)[i] << " ";
                    }
                    cout << endl;
                }
                cout << endl;

                // LoopID = geo_verification_icp(cand_list_2);
                LoopID = geo_verification_ransac(cand_list_2);

                gettimeofday(&t4, NULL);
                timeuse = (t4.tv_sec - t3.tv_sec) + (double)(t4.tv_usec - t3.tv_usec)/1000000.0;
                cout << "Registration time: " << timeuse << endl;
                RegistrationTime.push_back(timeuse);
                if(LoopID==-1){
                    NoLoopRegistrationTime.push_back(timeuse);
                }
                else{
                    WithLoopRegistrationTime.push_back(timeuse);
                }

                timeuse = (t4.tv_sec - t2.tv_sec) + (double)(t4.tv_usec - t2.tv_usec)/1000000.0;
                cout << "geo_verification time: " << timeuse << endl;
                GeoVerificationTime.push_back(timeuse);

                return LoopID;
            }
        }

    }

    vector<vector<int>> iBoW3D::cand_selection()
    {
        // int no_loop_flag = 1;

        // choose the non-zero part of current histogram
        vector<int> chosen_idx_non_zero;
        int non_zero_num;
        int i = 0;
        for(i=0; i < current_hist.rows(); i++)
        {
            if (current_hist(i)>0){
                chosen_idx_non_zero.push_back(i);
            }
        }
        non_zero_num = (int)(chosen_idx_non_zero.size());

        /* hist_cand saves all candidates;
           hist_idx_cand is the real frame idx, hist_idx_cand_DB is the idx of Database */
        vector<int> hist_idx_cand;
        vector<int> hist_idx_cand_DB;

        // coarse selection
        vector<Eigen::VectorXd>::iterator iter;
        Eigen::VectorXd temp_hist;
        vector<int> frameIdx = pDB->get_frameIdx();

        for(int i=0; i<(int)(hist_DB.size()); i++)
        {
            if(frameIdx[i] > frameID-gap_num)
            {
                break;
            }

            temp_hist = (hist_DB[i])(chosen_idx_non_zero, Eigen::all);

            if ( (temp_hist.array() > 0).select(1, temp_hist).sum() > lambda_word*non_zero_num )
            {
                hist_idx_cand.push_back(frameIdx[i]);
                hist_idx_cand_DB.push_back(i);
            }

        }


        if (hist_idx_cand.size()==0) // no scans after coarse selection
        {
            vector<vector<int>> island;
            return island;
        }
        else
        {
            // fine selection
            // calculate distances and save
            vector<int>::iterator iter;
            vector<pair<double, int>> distance_list;
            double dist1, dist;
            for(iter=hist_idx_cand_DB.begin(); iter!=hist_idx_cand_DB.end(); iter++)
            {
                dist1 = (current_hist-hist_DB[*iter]).norm();
                dist = dist1;
                distance_list.push_back(make_pair(dist, hist_idx_cand[iter-hist_idx_cand_DB.begin()]));
                // cout << dist << "/" << hist_idx_cand[iter-hist_idx_cand_DB.begin()] << "  ";
            }
            // sort distances
            sort(distance_list.begin(),distance_list.end(), [](const pair<double, int>& a, const pair<double, int>& b){ return a.first < b.first; });
            // distance_list.assign(distance_list.begin(), distance_list.begin() + min({distance_length, (int)(distance_list.size())}));

            vector<pair<double, int>>::iterator iter1, iter2;
            vector<vector<int>> idx_island_list;
            vector<double> avg_dist_list;
            vector<int> idx_num_list;
            vector<int> idx_island_temp;
            int chosen_old_loop_flag = 0;
            vector<int> island1, island2, island3;
            for(iter1=distance_list.begin(); iter1!=distance_list.end(); iter1++)
            {
                double avg_dist = mean_candidate_distance(distance_list, (*iter1).second, near_num, &idx_island_temp);
                int idx_num = static_cast<int>(idx_island_temp.size());
                idx_island_list.push_back(idx_island_temp);
                avg_dist_list.push_back(avg_dist);
                idx_num_list.push_back(idx_num);
                if (chosen_old_loop_flag==0 && abs(lastLoopID-(*iter1).second)<near_num/2)
                {
                    island3.assign(idx_island_temp.begin(), idx_island_temp.begin() + min({idx_num, search_num}));
                    chosen_old_loop_flag = 1;
                }
            }
            int max_num_idx = distance(idx_num_list.begin(), max_element(idx_num_list.begin(), idx_num_list.end()));
            island1.assign(idx_island_list[max_num_idx].begin(), idx_island_list[max_num_idx].begin() + min({idx_num_list[max_num_idx], search_num}));

            int min_dist_idx = distance(avg_dist_list.begin(), min_element(avg_dist_list.begin(), avg_dist_list.end()));
            island2.assign(idx_island_list[min_dist_idx].begin(), idx_island_list[min_dist_idx].begin() + min({idx_num_list[min_dist_idx], search_num}));

            vector<int> island22, island32;
            vector<int>::iterator find_it, find_it_2;
            for(int i=0; i<(int)(island2.size()); i++)
            {
                find_it = find(island1.begin(), island1.end(), island2[i]);
                if(find_it==island1.end()) { island22.push_back(island2[i]); }
            }
            if (chosen_old_loop_flag == 1) // there is island3
            {
                for(int i=0; i<(int)(island3.size()); i++)
                {
                    find_it = find(island1.begin(), island1.end(), island3[i]);
                    if(find_it==island1.end())
                    {
                        find_it_2 = find(island22.begin(), island22.end(), island3[i]);
                        if(find_it_2==island22.end()) { island32.push_back(island3[i]); }
                    }
                }
            }

            vector<vector<int>> cand_list;

            sort(island1.begin(),island1.end(), [](const int& a, const int& b){ return a < b; });
            if(island1.size() > 2)
            {
                swap(island1[0], island1[(int)(island1.size())/2]);
            }
            sort(island22.begin(),island22.end(), [](const int& a, const int& b){ return a < b; });
            if(island22.size() > 2)
            {
                swap(island22[0], island22[(int)(island22.size())/2]);
            }

            cand_list.push_back(island1);
            cand_list.push_back(island22);

            if (chosen_old_loop_flag == 1) // there is island3
            {
                sort(island32.begin(),island32.end(), [](const int& a, const int& b){ return a < b; });
                if(island32.size() > 2)
                {
                    swap(island32[0], island32[(int)(island32.size())/2]);
                }
                cand_list.push_back(island32);
            }

            // island1.insert(island1.end(), island2.begin(), island2.end());
            // if (chosen_old_loop_flag == 1) // there is island3
            // {
            //     island1.insert(island1.end(), island3.begin(), island3.end());
            // }
            // set<int> s(island1.begin(), island1.end());
            // island1.assign(s.begin(), s.end());

            return cand_list;
        }

    }


    // void iBoW3D::pcd_ransac(pair<int, reg_info> cand_info_pair, vector<result_struct> & res_list)
    // {
    //     pipelines::registration::RegistrationResult registration_result;

    //     bool mutual_filter = false;
    //     double distance_threshold = 5.0;
    //     int max_iteration = max_iter;

    //     // Prepare checkers
    //     std::vector<std::reference_wrapper<const pipelines::registration::CorrespondenceChecker>> correspondence_checker;
    //     auto correspondence_checker_edge_length = pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(0.9);
    //     auto correspondence_checker_distance = pipelines::registration::CorrespondenceCheckerBasedOnDistance(distance_threshold);
    //     correspondence_checker.push_back(correspondence_checker_edge_length);
    //     correspondence_checker.push_back(correspondence_checker_distance);

    //     registration_result = pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
    //             *(cand_info_pair.second.current_pcd), *(cand_info_pair.second.pCand), *(cand_info_pair.second.current_features), *(cand_info_pair.second.cand_features),
    //             mutual_filter, distance_threshold,
    //             pipelines::registration::TransformationEstimationPointToPoint(false),
    //             ransac_n, correspondence_checker,
    //             pipelines::registration::RANSACConvergenceCriteria(max_iteration, 0.999));

    //     res_list[cand_info_pair.first].fit = registration_result.fitness_;
    //     res_list[cand_info_pair.first].score = registration_result.inlier_rmse_;
    //     res_list[cand_info_pair.first].trans = registration_result.transformation_;
    //     res_list[cand_info_pair.first].idx = cand_info_pair.second.idx;

    // }

    pipelines::registration::RegistrationResult iBoW3D::run_registration(
            const geometry::PointCloud& source,
            const geometry::PointCloud& target,
            const pipelines::registration::Feature& source_feature,
            const pipelines::registration::Feature& target_feature)
    {
        double distance_threshold = 5.0;
        if(registration_backend == RegistrationBackend::FGR)
        {
            pipelines::registration::FastGlobalRegistrationOption option(
                    1.4, true, true, distance_threshold, max_iter, 0.95, 1000, true);
            return pipelines::registration::FastGlobalRegistrationBasedOnFeatureMatching(
                    source, target, source_feature, target_feature, option);
        }

        bool mutual_filter = false;
        int max_iteration = max_iter;

        std::vector<std::reference_wrapper<const pipelines::registration::CorrespondenceChecker>> correspondence_checker;
        auto correspondence_checker_edge_length = pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(0.9);
        auto correspondence_checker_distance = pipelines::registration::CorrespondenceCheckerBasedOnDistance(distance_threshold);
        correspondence_checker.push_back(correspondence_checker_edge_length);
        correspondence_checker.push_back(correspondence_checker_distance);

        open3d::utility::random::Seed(1);
        return pipelines::registration::RegistrationRANSACBasedOnFeatureMatching(
                source, target, source_feature, target_feature,
                mutual_filter, distance_threshold,
                pipelines::registration::TransformationEstimationPointToPoint(false),
                ransac_n, correspondence_checker,
                pipelines::registration::RANSACConvergenceCriteria(max_iteration, 0.999));
    }


    void iBoW3D::pcd_ransac(pair<int, int> cand_pair, vector<result_struct> & res_list,
                            std::shared_ptr<geometry::PointCloud> current_pcd,
                            std::shared_ptr<pipelines::registration::Feature> current_features)
    {
        int iter_find = pDB->get_DBIdx_by_idx(cand_pair.second);

        std::shared_ptr<geometry::PointCloud> pCand = pDB->get_cloud_by_idx(iter_find);
        cv::Mat cand_all_feat = pDB->get_all_feat_by_idx(iter_find);

        std::shared_ptr<pipelines::registration::Feature> cand_features(new pipelines::registration::Feature);
        int cand_size = int(pCand->points_.size());
        cand_features->Resize(feature_dim, cand_size);
        Eigen::MatrixXf cand_data_f(feature_dim, cand_size);
        Eigen::MatrixXd cand_data(feature_dim, cand_size);

        cv::cv2eigen(cand_all_feat, cand_data_f);
        cand_data = cand_data_f.cast<double>();
        cand_features->data_ = cand_data.transpose();

        pipelines::registration::RegistrationResult registration_result;

        registration_result = run_registration(*current_pcd, *pCand, *current_features, *cand_features);

        res_list[cand_pair.first].fit = registration_result.fitness_;
        res_list[cand_pair.first].score = registration_result.inlier_rmse_;
        res_list[cand_pair.first].trans = registration_result.transformation_;
        res_list[cand_pair.first].idx = cand_pair.second;
    }


    void iBoW3D::island_ransac(pair<int, vector<int>> island_pair, vector<result_struct> & res_list,
                               std::shared_ptr<geometry::PointCloud> current_pcd,
                               std::shared_ptr<pipelines::registration::Feature> current_features)
    {
        int iter_find = pDB->get_DBIdx_by_idx((island_pair.second)[0]);

        std::shared_ptr<geometry::PointCloud> pCand;
        std::shared_ptr<pipelines::registration::Feature> cand_features(new pipelines::registration::Feature);

        pCand = pDB->get_cloud_by_idx(iter_find);
        cv::Mat cand_all_feat = pDB->get_all_feat_by_idx(iter_find);

        int cand_size = int(pCand->points_.size());
        cand_features->Resize(feature_dim, cand_size);
        Eigen::MatrixXf cand_data_f(feature_dim, cand_size);
        Eigen::MatrixXd cand_data(feature_dim, cand_size);

        cv::cv2eigen(cand_all_feat, cand_data_f);
        cand_data = cand_data_f.cast<double>();
        cand_features->data_ = cand_data.transpose();

        // Perform alignment
        pipelines::registration::RegistrationResult registration_result;

        registration_result = run_registration(*current_pcd, *pCand, *current_features, *cand_features);

        if(registration_result.fitness_>fit_th_2 && registration_result.inlier_rmse_<score_th_2)
        {
            int current_res_list_idx = island_pair.first;
            res_list[current_res_list_idx].fit = registration_result.fitness_;
            res_list[current_res_list_idx].score = registration_result.inlier_rmse_;
            res_list[current_res_list_idx].trans = registration_result.transformation_;
            res_list[current_res_list_idx].idx = (island_pair.second)[0];

            vector<pair<int, int>> island_with_idx_list;
            for(int i=1; i<(int)((island_pair.second).size()); i++)
            {
                island_with_idx_list.push_back(make_pair(current_res_list_idx+i, (island_pair.second)[i]));
            }

            std::for_each(std::execution::par, island_with_idx_list.begin(), island_with_idx_list.end(),
                        [this, &res_list, current_pcd, current_features](pair<int, int> cand_pair)
                        { pcd_ransac(cand_pair, res_list, current_pcd, current_features); });

            // std::for_each(std::execution::par, cand_info_list.begin(), cand_info_list.end(),
            // [this, &res_list](pair<int, reg_info> cand_info_pair){ pcd_ransac(cand_info_pair, res_list); });
        }
    }

    int iBoW3D::geo_verification_ransac(vector<vector<int>> cand_list)
    {
        std::shared_ptr<pipelines::registration::Feature> current_features(new pipelines::registration::Feature);
        int current_size = int(current_pcd->points_.size());
        current_features->Resize(feature_dim, current_size);
        Eigen::MatrixXf current_data_f(feature_dim, current_size);
        Eigen::MatrixXd current_data(feature_dim, current_size);

        cv::Mat curr_all_feat = pFC->getAllFeature();

        cv::cv2eigen(curr_all_feat, current_data_f);
        current_data = current_data_f.cast<double>();
        current_features->data_ = current_data.transpose();

        vector<result_struct> res_list;
        vector<pair<int, vector<int>>> new_cand_list;

        int cand_list_len = 0;
        for(int i=0; i<(int)(cand_list.size()); i++)
        {
            new_cand_list.push_back(make_pair(cand_list_len, cand_list[i]));
            cand_list_len += (int)(cand_list[i].size());
        }
        for(int i=0; i<cand_list_len; i++)
        {
            result_struct res;
            res_list.push_back(res);
        }

        std::for_each(std::execution::par, new_cand_list.begin(), new_cand_list.end(),
                        [this, &res_list, current_features](pair<int, vector<int>> island_pair){ island_ransac(island_pair, res_list, this->current_pcd, current_features); });

        cout << "res_list: " << res_list.size() << endl;
        for(int i=0; i<(int)(res_list.size()); i++)
        {
            cout << "fitness: " << res_list[i].fit << " ";
            cout << "RMSE: " << res_list[i].score << " ";
            cout << "idx: " << res_list[i].idx << endl;
        }

        if (prior_fit)
        {
            // get the idx of highest fit, more fit is better; sort with fit in descending order

            sort(res_list.begin(),res_list.end(), [](const result_struct& a, const result_struct& b){ return a.fit > b.fit; });
            for(int i=0; i<(int)(res_list.size()); i++)
            {
                if( res_list[i].fit>fit_th && res_list[i].score<score_th )
                {
                    cout << "Fit:" << res_list[i].fit << endl;
                    cout << "Score:" << res_list[i].score << endl;
                    Eigen::Matrix4d_u transformation = res_list[i].trans;
                    trans = transformation;
                    lastLoopID = res_list[i].idx;
                    return lastLoopID;
                }
                else
                {
                    continue;
                }
            }
            return -1;
        }
        else
        {
            // get the idx of least score, less score is better; sort with score in ascending order

            sort(res_list.begin(),res_list.end(), [](const result_struct& a, const result_struct& b){ return a.score < b.score; });
            for(int i=0; i<(int)(res_list.size()); i++)
            {
                if( res_list[i].fit>fit_th && res_list[i].score<score_th )
                {
                    cout << "Fit:" << res_list[i].fit << endl;
                    cout << "Score:" << res_list[i].score << endl;
                    Eigen::Matrix4d_u transformation = res_list[i].trans;
                    trans = transformation;
                    lastLoopID = res_list[i].idx;
                    return lastLoopID;
                }
                else
                {
                    continue;
                }
            }
            return -1;
        }

    }

    vector<vector<int>> iBoW3D::range_check(vector<vector<int>> cand_list)
    {
        // Eigen::Vector3d max_point, min_point;
        // if(remove_outliers)
        // {
        //     auto current_pcd_tuple = current_pcd->RemoveRadiusOutliers(30,40);
        //     max_point = get<0>(current_pcd_tuple)->GetMaxBound();
        //     min_point = get<0>(current_pcd_tuple)->GetMinBound();
        // }
        // else
        // {
        //     max_point = current_pcd->GetMaxBound();
        //     min_point = current_pcd->GetMinBound();
        // }

        // double max_x_curr, min_x_curr, max_y_curr, min_y_curr;
        // max_x_curr = max_point[0];
        // min_x_curr = min_point[0];
        // max_y_curr = max_point[1];
        // min_y_curr = min_point[1];

        double max_x_curr, min_x_curr, max_y_curr, min_y_curr;
        vector<Eigen::Vector3d> current_pcd_points = current_pcd->points_;
        sort(current_pcd_points.begin(),current_pcd_points.end(), [](const Eigen::Vector3d & a, const Eigen::Vector3d & b){ return a[0] < b[0]; });
        max_x_curr = current_pcd_points[(int)(current_pcd_points.size()*0.995)][0];
        min_x_curr = current_pcd_points[(int)(current_pcd_points.size()*0.005)][0];
        sort(current_pcd_points.begin(),current_pcd_points.end(), [](const Eigen::Vector3d & a, const Eigen::Vector3d & b){ return a[1] < b[1]; });
        max_y_curr = current_pcd_points[(int)(current_pcd_points.size()*0.995)][1];
        min_y_curr = current_pcd_points[(int)(current_pcd_points.size()*0.005)][1];


        std::shared_ptr<geometry::PointCloud> pCand;
        double max_x_cand, min_x_cand, max_y_cand, min_y_cand;
        double max_x_rate, min_x_rate, max_y_rate, min_y_rate;

        vector<vector<int>> new_cand_list;

        Eigen::Vector3d max_point_2, min_point_2;
        vector<vector<int>>::iterator iter;
        for(iter=cand_list.begin(); iter!=cand_list.end(); iter++)
        {
            vector<int> new_island;
            for(int i=0; i<(int)((*iter).size()); i++)
            {
                int iter_find = pDB->get_DBIdx_by_idx((*iter)[i]);
                pCand = pDB->get_cloud_by_idx(iter_find);

                // if(remove_outliers)
                // {
                //     auto pCand_tuple = pCand->RemoveRadiusOutliers(30,40);
                //     max_point_2 = get<0>(pCand_tuple)->GetMaxBound();
                //     min_point_2 = get<0>(pCand_tuple)->GetMinBound();
                // }
                // else
                // {
                //     max_point_2 = pCand->GetMaxBound();
                //     min_point_2 = pCand->GetMinBound();
                // }

                // max_x_rate = max_point_2[0]/max_x_curr;
                // min_x_rate = min_point_2[0]/min_x_curr;
                // max_y_rate = max_point_2[1]/max_y_curr;
                // min_y_rate = min_point_2[1]/min_y_curr;

                vector<Eigen::Vector3d> pCand_points = pCand->points_;
                sort(pCand_points.begin(),pCand_points.end(), [](const Eigen::Vector3d & a, const Eigen::Vector3d & b){ return a[0] < b[0]; });
                max_x_cand = pCand_points[(int)(pCand_points.size()*0.995)][0];
                min_x_cand = pCand_points[(int)(pCand_points.size()*0.005)][0];
                sort(pCand_points.begin(),pCand_points.end(), [](const Eigen::Vector3d & a, const Eigen::Vector3d & b){ return a[1] < b[1]; });
                max_y_cand = pCand_points[(int)(pCand_points.size()*0.995)][1];
                min_y_cand = pCand_points[(int)(pCand_points.size()*0.005)][1];

                max_x_rate = max_x_cand/max_x_curr;
                min_x_rate = min_x_cand/min_x_curr;
                max_y_rate = max_y_cand/max_y_curr;
                min_y_rate = min_y_cand/min_y_curr;

                if(1/check_th < max_x_rate && max_x_rate < check_th &&
                   1/check_th < min_x_rate && min_x_rate < check_th &&
                   1/check_th < max_y_rate && max_y_rate < check_th &&
                   1/check_th < min_y_rate && min_y_rate < check_th)
                {
                    new_island.push_back((*iter)[i]);
                }
            }
            if(!new_island.empty()) // new_island is not empty
            {
                new_cand_list.push_back(new_island);
            }
        }

        return new_cand_list;
    }


}
