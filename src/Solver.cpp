#include "Solver.h"

namespace online_calibration
{
	Solver::Solver(std::string file_name_, int num_frames_, int num_cams_, CameraFrame* solver_frames_mp, Tracking* solver_track_mp)
		:solver_frames(solver_frames_mp), solver_track(solver_track_mp), 
        num_frames(num_frames_), num_cams(num_cams_), file_name(file_name_)
	{
        min_matches = 0; // minimum number of feature matches to proceed
        loop_close_thresh = 10; // meters
        match_thresh = 29; // bits, hamming distance for FREAK features
        f2f_iterations = 2;
        icp_iterations = 3;

        weight_3D2D = 10,
        weight_2D2D = 500,
        weight_3DPD = 1, // there are more of them
        loss_thresh_3D2D = 0.01, // reprojection error, canonical camera units
        loss_thresh_2D2D = 0.00002,
        loss_thresh_3DPD = 0.1, // physical distance, meters
        loss_thresh_3D3D = 0.04; // physical distance, meters
        outlier_reject = 5.0;

        correspondence_thresh_icp = 0.5,
        icp_norm_condition = 1e-5,
		agreement_t_thresh = 0.1, // meters
		agreement_r_thresh = 0.05; // radians

        icp_skip = 200,
		f2f_iterations = 2,
		icp_iterations = 3;
        ba_every = 10; // bundle adjust every this number of frames

        keypoint_obs_count_hist.resize(1000);

        
		ceres_poses_mat.resize(num_frames);
		ceres_poses_mat[0] = Eigen::Matrix4d::Identity();
		ceres_poses_vec = std::vector<double[6]>(num_frames);
		landmarks = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

        isam_slamInit();
        added_to_isam_3d = std::vector<std::vector<
        	std::map<int, isam::Pose3d_Point3d_Factor*>>>(num_cams,
            	std::vector<std::map<int, isam::Pose3d_Point3d_Factor*>>(num_frames));
		added_to_isam_2d = std::vector<std::vector<
			std::map<int, isam::Monocular_Factor*>>>(num_cams,
				std::vector<std::map<int, isam::Monocular_Factor*>>(num_frames));

        isam::Pose3d_Factor* prior2 = new isam::Pose3d_Factor(
            cam_nodes[0][0], origin, noiseless6);
        calib_isam->add_factor(prior2);
	}
	Solver::~Solver()
	{
	
	}
    void Solver::isam_add_node(int frame)
    {
        if(frame > 0) {
            for(int cam = 0; cam<num_cams; cam++) {
                calib_isam->add_node(cam_nodes[cam][frame]);
            }
        }
    }
    bool Solver::pose_init(const int ba, const int frame, const int dframe, double *transform)
    {
        //std::cout << "Solver::pose_init  test1  ceres_poses_mat.size = "<<(ceres_poses_mat.size())<<std::endl;
        //std::cout << "Solver::pose_init  frame = "<<(frame)<<std::endl;
        //std::cout << "Solver::pose_init  dframe = "<<(dframe)<<std::endl;
        //std::cout << "Solver::pose_init  ba = "<<(ba)<<std::endl;
        if(frame-dframe < 0) 
            return false;
        // matches are what's fed into frameToFrame,
        // good matches have outliers removed during optimization
        //std::cout << "Solver::pose_init  test2 "<<std::endl;
        Eigen::Matrix4d dT;
        if(dframe == 1) {
            if(frame > 1) {
                //Eigen::Matrix4d T1 = cam_nodes[0][frame-2]->value().wTo();
                //Eigen::Matrix4d T2 = cam_nodes[0][frame-1]->value().wTo();
                auto T1 = ceres_poses_mat[frame-2];
                auto T2 = ceres_poses_mat[frame-1];
                dT = T1.inverse() * T2;
            } else {
                dT = Tools::pose_mat2vec(transform);
            }
        } else {
            //Eigen::Matrix4d T1 = cam_nodes[0][frame-dframe]->value().wTo();
            //Eigen::Matrix4d T2 = cam_nodes[0][frame]->value().wTo();
            auto T1 = ceres_poses_mat[frame-dframe];
            auto T2 = ceres_poses_mat[frame];
            dT = T1.inverse() * T2;
        }
        if(ba == 1) 
            dT(1,3) /= 20;
        Tools::pose_vec2mat(dT, transform);
        return true;
    }
    bool Solver::loopclosure_compute(const int ba, double *transform)
    {
        // if attempting loop closure,
        // check if things are close by
        if(ba == 1) {
            Eigen::Vector3d le_dist; le_dist << transform[3], transform[4], transform[5];
            if(le_dist.norm() > loop_close_thresh) {
                return false;
            } else {
                std::cerr << "Loop closure plausible" << std::endl;
                return true;
            }
        }
        return true;
    }
    void Solver::isam_slamInit()
    {

        calib_isam = new isam::Slam();
        isam::Properties prop = calib_isam->properties();
        prop.max_iterations = 50;
        calib_isam->set_properties(prop);
        cam_nodes = std::vector<std::vector<isam::Pose3d_Node*>>(num_cams);
        monoculars = std::vector<isam::MonocularCamera>(num_cams);
        for(int cam=0; cam<num_cams; cam++) {
            monoculars[cam] = isam::MonocularCamera(1, Eigen::Vector2d(0, 0));
            for(int frame = 0; frame < num_frames; frame++) {
                isam::Pose3d_Node* initial_node = new isam::Pose3d_Node();
                cam_nodes[cam].push_back(initial_node);
            }
            calib_isam->add_node(cam_nodes[cam][0]);
        }
        
        noiseless6 = isam::Information(1000. * isam::eye(6));
        noisy6 = isam::Information(1 * isam::eye(6));
        isam::Pose3d origin;
        isam::Pose3d_Factor* prior = new isam::Pose3d_Factor(
                cam_nodes[0][0], origin, noiseless6);
        calib_isam->add_factor(prior);
        for(int cam = 1; cam<num_cams; cam++) {
            isam::Pose3d_Pose3d_Factor* cam_factor = new isam::Pose3d_Pose3d_Factor(
                    cam_nodes[0][0],
                    cam_nodes[cam][0],
                    isam::Pose3d(solver_track->cam_pose[cam].cast<double>()),
                    noiseless6
                    );
            calib_isam->add_factor(cam_factor);
        }
    }
	
    void Solver::matchUsingId(
        const std::vector<std::vector<std::vector<int>>> &keypoint_ids,
        const int cam1,
        const int cam2,
        const int frame1,
        const int frame2,
        std::vector<std::pair<int, int>> &matches
        ) {
        std::map<int, int> id2ind;
        for(int ind = 0; ind < keypoint_ids[cam1][frame1].size(); ind++) {
            int id = keypoint_ids[cam1][frame1][ind];
            id2ind[id] = ind;
        }
        for(int ind = 0; ind < keypoint_ids[cam2][frame2].size(); ind++) {
            int id = keypoint_ids[cam2][frame2][ind];
            if(id2ind.count(id))
                matches.push_back(std::make_pair(id2ind[id], ind));
        }
    }
    void Solver::matchUsingId(
            const std::vector<std::vector<std::vector<int>>> &keypoint_ids,
            const int frame1,
            const int frame2,
            std::vector<std::vector<std::pair<int, int>>> &matches,
            int num_cams
            ) {
        for(int cam=0; cam<num_cams; cam++) {
            matchUsingId(keypoint_ids, cam, cam, frame1, frame2, matches[cam]);
        }
    }
    void Solver::matchFeatures(
        const std::vector<std::vector<cv::Mat>> &descriptors,
        const int cam1,
        const int cam2,
        const int frame1,
        const int frame2,
        std::vector<std::pair<int, int>> &matches
        ) {
        double start = clock()/double(CLOCKS_PER_SEC);
        /*
        std::cerr << "Matching: ";
        std::cerr << descriptors[cam1].size()
            << ", " << descriptors[cam2].size()
            << ", " << frame1 << " " << frame2 << "; ";
        std::cerr << descriptors[cam1][frame1].size()
            << ", " << descriptors[cam2][frame2].size();
            */
        std::vector<cv::DMatch> mc;
#ifdef USE_CUDA
        cv::Ptr<cv::cuda::DescriptorMatcher> d_matcher =
            cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
        cv::cuda::GpuMat d_query(descriptors[cam1][frame1]);
        cv::cuda::GpuMat d_train(descriptors[cam2][frame2]);
        cv::cuda::GpuMat d_matches;
        d_matcher->matchAsync(d_query, d_train, d_matches);

        d_matcher->matchConvert(d_matches, mc);
#else
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.match(
                descriptors[cam1][frame1],
                descriptors[cam2][frame2], mc);
#endif

        double end = clock()/double(CLOCKS_PER_SEC);
        //std::cerr << "; " << end-start << std::endl;
        // find minimum matching distance and filter out the ones more than twice as big as it
        double min_dist = 1e9, max_dist = 0;
        for(int i=0; i<mc.size(); i++) {
            if(mc[i].distance < min_dist) min_dist = mc[i].distance;
            if(mc[i].distance > max_dist) max_dist = mc[i].distance;
        }
        /*
        std::cerr << "Matches cam " << cam1 << ": " <<  mc.size()
            << " " << min_dist << " " << max_dist
            << std::endl;
            */
        for(int i=0; i<mc.size(); i++) {
            if(mc[i].distance > std::max(1.5*min_dist, match_thresh)) continue;
            matches.push_back(std::make_pair(mc[i].queryIdx, mc[i].trainIdx));
        }
    }
    void Solver::matchFeatures(
            const std::vector<std::vector<cv::Mat>> &descriptors,
            const int frame1,
            const int frame2,
            std::vector<std::vector<std::pair<int, int>>> &matches
            ) {
        for(int cam=0; cam<num_cams; cam++) {
            matchFeatures(descriptors, cam, cam, frame1, frame2, matches[cam]);
        }
    }
    void Solver::getLandmarksAtFrame(
        //const Eigen::Matrix4d &pose,
        //const pcl::PointCloud<pcl::PointXYZ>::Ptr landmarks,
        //const std::vector<bool> &keypoint_added,
        //const std::vector<std::vector<std::vector<int>>> &keypoint_ids,
        const int frame,
        const int dframe,
        std::map<int, pcl::PointXYZ> &landmarks_at_frame)
    {
        Eigen::Matrix4d poseinv = ceres_poses_mat[frame-dframe].inverse();
        for(int cam = 0; cam < num_cams; cam++) {
            for(int id : solver_frames->keypoint_ids[cam][frame]) {
                if(landmarks_at_frame.count(id)) continue;
                if(!keypoint_added[id]) continue;
                auto point = landmarks->at(id);
                Eigen::Vector4d q;
                q << point.x, point.y, point.z, 1;
                Eigen::Vector4d p = poseinv * q;
                pcl::PointXYZ pp;
                pp.x = p(0)/p(3);
                pp.y = p(1)/p(3);
                pp.z = p(2)/p(3);
                landmarks_at_frame[id] = pp;
                /*
                std::cerr << "Getting landmark " << id << ": "
                    << pp << std::endl;
                std::cerr << pose << std::endl;
                */
            }
        }
    }
    Eigen::Matrix4d Solver::frameToFrame(
    const std::vector<std::vector<std::pair<int, int>>> &matches,                   //特征点匹配关系
    //const std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints,            //特征点
    //const std::vector<std::vector<std::vector<int>>> &keypoint_ids,                 //特征点id
    const std::map<int, pcl::PointXYZ> &landmarks_at_frame,                         //已有的路标点
    //const std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> &keypoints_with_depth,
    //const std::vector<std::vector<std::vector<int>>> &has_depth,
    //const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans_M,                //当前帧激光数据
    //const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans_S,                //上一帧激光数据
    //const std::vector<pcl::KdTreeFLANN<pcl::PointXYZ>> &kd_trees,                   //上一帧激光点云
    const int frame1,                                                               //当前帧序号
    const int frame2,                                                               //上一帧序号
    double transform[6],
    std::vector<std::vector<std::pair<int, int>>> &good_matches,                    //优秀的匹配
    std::vector<std::vector<ResidualType>> &residual_type,                          //输出结果
    const bool enable_icp                                                           //是否使用icp
    )
    {
        for(int iter = 1; iter <= f2f_iterations; iter++) 
        {
            ceres::Problem::Options problem_options;
            problem_options.enable_fast_removal = true;
            ceres::Problem problem(problem_options);

            // Visual odometry
            for(int cam = 0; cam<num_cams; cam++) {
                //std::cerr << "Matches: " << matches[cam].size() << std::endl;
                good_matches[cam].clear();
                residual_type[cam].clear();
                const std::vector<std::pair<int, int>> &mc = matches[cam];
                for(int i=0; i<mc.size(); i++) {
                    int point1 = mc[i].first,
                        point2 = mc[i].second;
                    int id = solver_frames->keypoint_ids[cam][frame2][point2];
                    bool d1 = solver_frames->has_depth[cam][frame1][point1] != -1,
                        d2 = solver_frames->has_depth[cam][frame2][point2] != -1;
                    pcl::PointXYZ point3_2, point3_1;
                    if(landmarks_at_frame.count(id)) {
                        point3_2 = landmarks_at_frame.at(id);
                        /*
                        if(d2) {
                            std::cerr << "Using landmark "
                                << id << ": " << point3_2 
                                << " " << keypoints_with_depth[cam][frame2]
                                ->at(has_depth[cam][frame2][point2]) << std::endl;
                        }
                        */
                        d2 = true;
                    } else if(d2) {
                        point3_2 = solver_frames->kp_with_depth[cam][frame2]
                            ->at(solver_frames->has_depth[cam][frame2][point2]);
                    }
                    if(d1) {
                        point3_1 = solver_frames->kp_with_depth[cam][frame1]
                            ->at(solver_frames->has_depth[cam][frame1][point1]);
                    }
                    cv::Point2f point2_1 = solver_frames->keypoints[cam][frame1][point1];
                    cv::Point2f point2_2 = solver_frames->keypoints[cam][frame2][point2];
                    //std::cerr << "has depth: " << has_depth[cam][frame1].size();

                    //std::cerr << " " << has_depth[cam][frame1][point1]
                    //    << " " << keypoints_with_depth[cam][frame1]->size();
                    //std::cerr << " " << has_depth[cam][frame2][point2]
                    //    << " " << keypoints_with_depth[cam][frame2]->size();
                    //std::cerr << std::endl;
                    if(d1 && d2) {
                        // 3D 3D
                        cost3D3D *cost = new cost3D3D(
                                point3_1.x,
                                point3_1.y,
                                point3_1.z,
                                point3_2.x,
                                point3_2.y,
                                point3_2.z
                                );
                        double residual_test[3];
                        (*cost)(transform, residual_test);
                        if(iter > 1 &&
                                residual_test[0] * residual_test[0] +
                                residual_test[1] * residual_test[1] +
                                residual_test[2] * residual_test[2]
                                >
                                loss_thresh_3D3D*outlier_reject/iter *
                                loss_thresh_3D3D*outlier_reject/iter) {
                            continue;
                        }
                        ceres::CostFunction* cost_function =
                            new ceres::AutoDiffCostFunction<cost3D3D,3,6>(
                                    cost);
                        problem.AddResidualBlock(
                                cost_function,
                                new ceres::ArctanLoss(loss_thresh_3D3D),
                                transform);

                        residual_type[cam].push_back(RESIDUAL_3D3D);
                        good_matches[cam].push_back(std::make_pair(point1, point2));
                    }
                    if(!d1 && !d2) {
                        // 2D 2Dresidual_type
                        cost2D2D *cost =
                            new cost2D2D(
                                    point2_1.x,
                                    point2_1.y,
                                    point2_2.x,
                                    point2_2.y,
                                    solver_track->cam_trans[cam](0),
                                    solver_track->cam_trans[cam](1),
                                    solver_track->cam_trans[cam](2)
                                    );
                        double residual_test[1];
                        (*cost)(transform, residual_test);
                        if(iter > 1 && abs(residual_test[0]) > loss_thresh_2D2D*outlier_reject/iter) continue;
                        ceres::CostFunction* cost_function =
                            new ceres::AutoDiffCostFunction<cost2D2D,1,6>(cost);
                        problem.AddResidualBlock(
                                cost_function,
                                new ceres::ScaledLoss(
                                    new ceres::ArctanLoss(loss_thresh_2D2D),
                                    weight_2D2D,
                                    ceres::TAKE_OWNERSHIP),
                                transform);
                        residual_type[cam].push_back(RESIDUAL_2D2D);
                        good_matches[cam].push_back(std::make_pair(point1, point2));
                    }
                    if(d1) {
                        // 3D 2D
                        cost3D2D *cost =
                            new cost3D2D(
                                    point3_1.x,
                                    point3_1.y,
                                    point3_1.z,
                                    point2_2.x,
                                    point2_2.y,
                                    solver_track->cam_trans[cam](0),
                                    solver_track->cam_trans[cam](1),
                                    solver_track->cam_trans[cam](2)
                                    );
                        double residual_test[2];
                        (*cost)(transform, residual_test);
                        if(iter > 1 && residual_test[0] * residual_test[0] +
                                residual_test[1] * residual_test[1]
                                > loss_thresh_3D2D*outlier_reject/iter
                                * loss_thresh_3D2D*outlier_reject/iter) continue;

                        ceres::CostFunction* cost_function =
                            new ceres::AutoDiffCostFunction<cost3D2D,2,6>(cost);
                        problem.AddResidualBlock(
                                cost_function,
                                new ceres::ScaledLoss(
                                    new ceres::ArctanLoss(loss_thresh_3D2D),
                                    weight_3D2D,
                                    ceres::TAKE_OWNERSHIP),
                                transform);

                        residual_type[cam].push_back(RESIDUAL_3D2D);
                        good_matches[cam].push_back(std::make_pair(point1, point2));
                    }
                    if(d2) {
                        // 2D 3D
                        cost2D3D *cost =
                            new cost2D3D(
                                    point3_2.x,
                                    point3_2.y,
                                    point3_2.z,
                                    point2_1.x,
                                    point2_1.y,
                                    solver_track->cam_trans[cam](0),
                                    solver_track->cam_trans[cam](1),
                                    solver_track->cam_trans[cam](2)
                                    );
                        double residual_test[2];
                        (*cost)(transform, residual_test);
                        if(iter > 1 && residual_test[0] * residual_test[0] +
                                residual_test[1] * residual_test[1]
                                > loss_thresh_3D2D*outlier_reject/iter
                                * loss_thresh_3D2D*outlier_reject/iter) continue;

                        ceres::CostFunction* cost_function =
                            new ceres::AutoDiffCostFunction<cost2D3D,2,6>(cost);
                        problem.AddResidualBlock(
                                cost_function,
                                new ceres::ScaledLoss(
                                    new ceres::ArctanLoss(loss_thresh_3D2D),
                                    weight_3D2D,
                                    ceres::TAKE_OWNERSHIP),
                                transform);

                        residual_type[cam].push_back(RESIDUAL_2D3D);
                        good_matches[cam].push_back(std::make_pair(point1, point2));
                    }
                }
            }
            //Lidar odometry---- Point set registration
            // if(enable_icp)
            // {
            //     std::vector<ceres::ResidualBlockId> icp_blocks;

            //     //std::cerr << "M: " << scans_M.size() << " S: " << scans_S.size() << kd_trees.size() << std::endl;

            //     for(int icp_iter = 0; icp_iter < icp_iterations; icp_iter++) {
            //         while(icp_blocks.size() > 0) {
            //             auto bid = icp_blocks.back();
            //             icp_blocks.pop_back();
            //             problem.RemoveResidualBlock(bid);
            //         }
            //         for(int sm = 0; sm < scans_M.size() * enable_icp; sm++) {
            //             for(int smi = 0; smi < scans_M[sm]->size(); smi+= icp_skip) {
            //                 pcl::PointXYZ pointM = scans_M[sm]->at(smi);
            //                 pcl::PointXYZ pointM_untransformed = pointM;
            //                 Tools::transform_point(pointM, transform);
            //                 /*
            //                 * Point-to-plane ICP where plane is defined by
            //                 * three Nearest Points (np):
            //                 *            np_i     np_k
            //                 * np_s_i ..... * ..... * .....
            //                 *               \     /
            //                 *                \   /
            //                 *                 \ /
            //                 * np_s_j ......... * .......
            //                 *                 np_j
            //                 */
            //                 int np_i = 0, np_j = 0, np_k = 0;
            //                 int np_s_i = -1, np_s_j = -1;
            //                 double np_dist_i = INF, np_dist_j = INF;
            //                 for(int ss = 0; ss < kd_trees.size(); ss++) {
            //                     std::vector<int> id(1);
            //                     std::vector<float> dist2(1);
            //                     if(kd_trees[ss].nearestKSearch(pointM, 1, id, dist2) <= 0 ||
            //                             dist2[0] > correspondence_thresh_icp/iter/iter/iter/iter) {
            //                         continue;
            //                     }
            //                     pcl::PointXYZ np = scans_S[ss]->at(id[0]);

            //                     Tools::subtract_assign(np, pointM);
            //                     double d = Tools::norm2(np);
            //                     if(d < np_dist_i) {
            //                         np_dist_j = np_dist_i;
            //                         np_j = np_i;
            //                         np_s_j = np_s_i;
            //                         np_dist_i = d;
            //                         np_i = id[0];
            //                         np_s_i = ss;
            //                     } else if(d < np_dist_j) {
            //                         np_dist_j = d;
            //                         np_j = id[0];
            //                         np_s_j = ss;
            //                     }
            //                 }
            //                 if(np_s_i == -1 || np_s_j == -1) {
            //                     continue;
            //                 }
            //                 int np_k_n = scans_S[np_s_i]->size(),
            //                     np_k_1p = (np_i+1) % np_k_n,
            //                     np_k_2p = (np_i-1 + np_k_n) % np_k_n;
            //                 pcl::PointXYZ np_k_1 = scans_S[np_s_i]->at(np_k_1p),
            //                     np_k_2 = scans_S[np_s_i]->at(np_k_2p);
            //                 Tools::subtract_assign(np_k_1, pointM);
            //                 Tools::subtract_assign(np_k_2, pointM);
            //                 if(Tools::norm2(np_k_1) < Tools::norm2(np_k_2)) {
            //                     np_k = np_k_1p;
            //                 } else {
            //                     np_k = np_k_2p;
            //                 }
            //                 pcl::PointXYZ s0, s1, s2;
            //                 s0 = scans_S[np_s_i]->at(np_i);
            //                 s1 = scans_S[np_s_j]->at(np_j);
            //                 s2 = scans_S[np_s_i]->at(np_k);
            //                 Eigen::Vector3f
            //                     v0 = s0.getVector3fMap(),
            //                     v1 = s1.getVector3fMap(),
            //                     v2 = s2.getVector3fMap();
            //                 Eigen::Vector3f N = (v1 - v0).cross(v2 - v0);
            //                 if(N.norm() < icp_norm_condition) continue;
            //                 N /= N.norm();
            //                 ceres::CostFunction* cost_function =
            //                     new ceres::AutoDiffCostFunction<cost3DPD, 1, 6>(
            //                             new cost3DPD(
            //                                 pointM_untransformed.x,
            //                                 pointM_untransformed.y,
            //                                 pointM_untransformed.z,
            //                                 N[0], N[1], N[2],
            //                                 v0[0], v0[1], v0[2]
            //                                 )
            //                             );
            //                 auto bid = problem.AddResidualBlock(
            //                         cost_function,
            //                         new ceres::ScaledLoss(
            //                             new ceres::CauchyLoss(loss_thresh_3DPD),
            //                             weight_3DPD,
            //                             ceres::TAKE_OWNERSHIP),
            //                         transform);
            //                 icp_blocks.push_back(bid);
            //             }
            //         }
            //         //问题优化步骤，每次ICP迭代都需要计算
            // }
            //residualStats(problem, good_matches, residual_type);
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.minimizer_progress_to_stdout = false;
            //options.num_threads = 8;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            if(f2f_iterations - iter == 0) {
                //residualStats(problem, good_matches, residual_type);
            }
            residualStats(problem, good_matches, residual_type);
        }
        /*
       for(int i=0; i<6; i++) {
       std::cerr << transform[i] << " ";
       }
       std::cerr << std::endl;
       */
        return Tools::pose_mat2vec(transform);
    }
    void Solver::residualStats(
        ceres::Problem &problem,
        const std::vector<std::vector<std::pair<int, int>>> &good_matches,
        const std::vector<std::vector<ResidualType>> &residual_type
        )
    {
        // compute some statistics about residuals
        double cost;
        std::vector<double> residuals;
        ceres::Problem::EvaluateOptions evaluate_options;
        evaluate_options.apply_loss_function = false;
        problem.Evaluate(evaluate_options, &cost, &residuals, NULL, NULL);
        std::vector<double> residuals_3D3D, residuals_3D2D, residuals_2D3D, residuals_2D2D, residuals_3DPD;
        int ri = 0;
        for(int cam = 0; cam < num_cams; cam++) {
            auto &mc = good_matches[cam];
            for(int i=0; i<mc.size(); i++) {
                switch(residual_type[cam][i]) {
                    case RESIDUAL_3D3D:
                        residuals_3D3D.push_back(
                                sqrt(
                                    residuals[ri]*residuals[ri] +
                                    residuals[ri+1]*residuals[ri+1] +
                                    residuals[ri+2]*residuals[ri+2])
                                );
                        ri += 3;
                        break;
                    case RESIDUAL_3D2D:
                        residuals_3D2D.push_back(
                                sqrt(
                                    residuals[ri]*residuals[ri] +
                                    residuals[ri+1]*residuals[ri+1]
                                    )
                                );
                        ri += 2;
                        break;
                    case RESIDUAL_2D3D:
                        residuals_2D3D.push_back(
                                sqrt(
                                    residuals[ri]*residuals[ri] +
                                    residuals[ri+1]*residuals[ri+1]
                                    )
                                );
                        ri += 2;
                        break;
                    case RESIDUAL_2D2D:
                        residuals_2D2D.push_back(abs(residuals[ri]));
                        ri++;
                        break;
                    default:
                        break;
                }
            }
        }
        for(; ri < residuals.size(); ri++) {
            residuals_3DPD.push_back(abs(residuals[ri]));
        }
        double sum_3D3D = 0, sum_3D2D = 0, sum_2D3D = 0, sum_2D2D = 0, sum_3DPD = 0;
        for(auto r : residuals_3D3D) {sum_3D3D += r;}
        for(auto r : residuals_3D2D) {sum_3D2D += r;}
        for(auto r : residuals_2D3D) {sum_2D3D += r;}
        for(auto r : residuals_2D2D) {sum_2D2D += r;}
        for(auto r : residuals_3DPD) {sum_3DPD += r;}
        std::sort(residuals_3D3D.begin(), residuals_3D3D.end());
        std::sort(residuals_3D2D.begin(), residuals_3D2D.end());
        std::sort(residuals_2D3D.begin(), residuals_2D3D.end());
        std::sort(residuals_2D2D.begin(), residuals_2D2D.end());
        std::sort(residuals_3DPD.begin(), residuals_3DPD.end());
        std::cerr << "Cost: " << cost
            << " Residual blocks: " << problem.NumResidualBlocks()
            << " Residuals: " << problem.NumResiduals() << " " << ri << " " << residuals.size()
            << std::endl
            << " Total good matches: ";
        for(int cam=0; cam<num_cams; cam++) {
            std::cerr << " " << good_matches[cam].size();
        }
        for(int cam=0; cam<num_cams; cam++) {
            std::cerr << " " << residual_type[cam].size();
        }
        std::cerr << std::endl;
        if(residuals_3D3D.size())
            std::cerr << "Residual 3D3D:"
                << " median " << std::fixed << std::setprecision(10) << residuals_3D3D[residuals_3D3D.size()/2]
                << " mean " << std::fixed << std::setprecision(10) << sum_3D3D/residuals_3D3D.size()
                << " count " << residuals_3D3D.size() << std::endl;
        if(residuals_3D2D.size())
            std::cerr << "Residual 3D2D:"
                << " median " << std::fixed << std::setprecision(10) << residuals_3D2D[residuals_3D2D.size()/2]
                << " mean " << std::fixed << std::setprecision(10) << sum_3D2D/residuals_3D2D.size()
                << " count " << residuals_3D2D.size() << std::endl;
        if(residuals_2D3D.size())
            std::cerr << "Residual 2D3D:"
                << " median " << std::fixed << std::setprecision(10) << residuals_2D3D[residuals_2D3D.size()/2]
                << " mean " << std::fixed << std::setprecision(10) << sum_2D3D/residuals_2D3D.size()
                << " count " << residuals_2D3D.size() << std::endl;
        if(residuals_2D2D.size())
            std::cerr << "Residual 2D2D:"
                << " median " << std::fixed << std::setprecision(10) << residuals_2D2D[residuals_2D2D.size()/2]
                << " mean " << std::fixed << std::setprecision(10) << sum_2D2D/residuals_2D2D.size()
                << " count " << residuals_2D2D.size() << std::endl;
        if(residuals_3DPD.size())
            std::cerr << "Residual 3DPD:"
                << " median " << std::fixed << std::setprecision(10) << residuals_3DPD[residuals_3DPD.size()/2]
                << " mean " << std::fixed << std::setprecision(10) << sum_3DPD/residuals_3DPD.size()
                << " count " << residuals_3DPD.size() << std::endl;

    }
    void Solver::iSAM_update(int frame, int dframe, Eigen::Matrix4d& dpose)
    {
        // iSAM time!
        isam::Pose3d_Pose3d_Factor* odom_factor =
            new isam::Pose3d_Pose3d_Factor(
                    cam_nodes[0][frame-dframe],
                    cam_nodes[0][frame],
                    isam::Pose3d(dpose),
                    noisy6
                    );
        calib_isam->add_factor(odom_factor);
        for(int cam = 1; cam<num_cams; cam++) {
            isam::Pose3d_Pose3d_Factor* cam_factor =
                new isam::Pose3d_Pose3d_Factor(
                        cam_nodes[0][frame],
                        cam_nodes[cam][frame],
                        isam::Pose3d(solver_track->cam_pose[cam].cast<double>()),
                        noiseless6
                        );
            calib_isam->add_factor(cam_factor);
        }
        calib_isam->update();
    }
    void Solver::iSAM_add_keypoint(const int frame, int& id_counter, bool enable_isam)
    {
        if(enable_isam)
        {
            while(point_nodes.size() <= id_counter) {
                isam::Point3d_Node* point_node = new isam::Point3d_Node();
                point_nodes.push_back(point_node);
            }
        }
        keypoint_added.resize(id_counter+1, false);
        landmarks->resize(id_counter+1);
        keypoint_obs_count_hist[0] += id_counter+1 - solver_frames->keypoint_obs_count.size();
        solver_frames->keypoint_obs_count.resize(id_counter+1, 0);
        solver_frames->keypoint_obs2.resize(id_counter+1,
                std::vector<std::map<int, cv::Point2f>>(num_cams));
        solver_frames->keypoint_obs3.resize(id_counter+1,
                std::vector<std::map<int, pcl::PointXYZ>>(num_cams));
        for(int cam=0; cam<num_cams; cam++) {
            for(int i=0; i<solver_frames->keypoints[cam][frame].size(); i++) {
                int id = solver_frames->keypoint_ids[cam][frame][i];
                solver_frames->keypoint_obs_count[id]++;
                int koc = solver_frames->keypoint_obs_count[id];
                if(koc >= keypoint_obs_count_hist.size()) {
                    keypoint_obs_count_hist.resize(koc+1, 0);
                }
                keypoint_obs_count_hist[koc-1]--;
                keypoint_obs_count_hist[koc]++;
                /*
                std::cerr << i << "," << has_depth[cam][frame][i] 
                << "/" << keypoints[cam][frame].size()
                << " " << std::endl;
                */
                if(solver_frames->has_depth[cam][frame][i] == -1) {
                    solver_frames->keypoint_obs2[id][cam][frame] = solver_frames->keypoints[cam][frame][i];
                } else {
                    solver_frames->keypoint_obs3[id][cam][frame] = solver_frames->kp_with_depth[cam][frame]->at(
                            solver_frames->has_depth[cam][frame][i]);
                }
            }
            //std::cerr << std::endl;
        }
        std::cerr << "Keypoint observation stats: ";
        for(int i=0; i<10; i++) {
            std::cerr << keypoint_obs_count_hist[i] << " ";
        }
        int zxcv = 0; for(auto h : keypoint_obs_count_hist) zxcv += h;
        std::cerr << zxcv << std::endl;
        std::cerr << "Features: " << id_counter+1 << std::endl;
    }
    void Solver::iSAM_add_measurement(const int frame)
    {
        std::cout << "Solver::iSAM_add_measurement  test0"<<std::endl;
        std::set<int> ids_seen;
        for(int cam=0; cam<num_cams; cam++) {
            for(int i=0; i<solver_frames->keypoints[cam][frame].size(); i++) {
                int id = solver_frames->keypoint_ids[cam][frame][i];
                ids_seen.insert(id);
            }
        }
        for(auto id : ids_seen) {
            if(solver_frames->keypoint_obs_count[id] < 3) {
                continue;
            }
            triangulatePoint(id);
            if(!keypoint_added[id]) {
                calib_isam->add_node(point_nodes[id]);
                keypoint_added[id] = true;
            }
            for(int cam=0; cam<num_cams; cam++) {
                for(auto obs3 : solver_frames->keypoint_obs3[id][cam]) {
                    if(added_to_isam_3d[cam][obs3.first].count(id)) {
                        continue;
                    }
                    isam::Noise noise3 = isam::Information(1 * isam::eye(3));
                    auto p3 = obs3.second;
                    Eigen::Vector3d point3d = p3.getVector3fMap().cast<double>();
                    isam::Pose3d_Point3d_Factor* factor =
                        new isam::Pose3d_Point3d_Factor(
                                // 3D points always in cam 0 frame
                                cam_nodes[0][obs3.first], 
                                point_nodes[id],
                                isam::Point3d(point3d),
                                noise3
                                );
                    calib_isam->add_factor(factor);
                    added_to_isam_3d[cam][obs3.first][id] = factor;
                }
            }
            std::cout << "Solver::iSAM_add_measurement  test1"<<std::endl;
            for(int cam=0; cam<num_cams; cam++) {
                for(auto obs2 : solver_frames->keypoint_obs2[id][cam]) {
                    if(added_to_isam_2d[cam][obs2.first].count(id)) {
                        continue;
                    }
                    isam::MonocularMeasurement measurement(
                            obs2.second.x,
                            obs2.second.y
                            );
                    isam::Noise noise2 = isam::Information(1 * isam::eye(2));
                    std::cout << "Solver::iSAM_add_measurement  test1.3 cam_node = "<<(*(cam_nodes[cam][obs2.first]))<<std::endl;
                    std::cout << "Solver::iSAM_add_measurement  cam = "<<(cam)<<std::endl;
                    std::cout << "Solver::iSAM_add_measurement  obs2.first = "<<(obs2.first)<<std::endl;
                    isam::Monocular_Factor* factor =
                        new isam::Monocular_Factor(
                                cam_nodes[cam][obs2.first],
                                point_nodes[id],
                                &(monoculars[cam]),
                                measurement,
                                noise2
                                );
                    std::cout << "Solver::iSAM_add_measurement  test1.4"<<std::endl;
                    calib_isam->add_factor(factor);
                    std::cout << "Solver::iSAM_add_measurement  test1.5"<<std::endl;
                    added_to_isam_2d[cam][obs2.first][id] = factor;
                    std::cout << "Solver::iSAM_add_measurement  test1.7"<<std::endl;
                }
            }
            std::cout << "Solver::iSAM_add_measurement  test2"<<std::endl;
        }

    }
    void Solver::triangulatePoint(const int id)
    {
        bool initial_guess = keypoint_added[id];
        pcl::PointXYZ &point = landmarks->at(id);
        const std::vector<double[6]> &camera_poses = ceres_poses_vec;
        // given the 2D and 3D observations, the goal is to obtain
        // the 3D position of the point
        int initialized = 0;
        // 0: uninitialized
        // 1: one 2d measurement
        // 2: two 2d measurements
        // 3: initialized
        double transform[3] = {0, 0, 10};
        if(initial_guess) {
            transform[0] = point.x;
            transform[1] = point.y;
            transform[2] = point.z;
            initialized = 3;
        }

        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        for(int cam=0; cam<num_cams; cam++) {
            for(auto obs3 : solver_frames->keypoint_obs3[id][cam]) {
                /*
                std::cerr << "3D observation " <<  obs3.second
                    << " at " << obs3.first << ": ";
                for(int i=0; i<6; i++) {
                    std::cerr << camera_poses[obs3.first][i] << " ";
                }
                std::cerr << std::endl;
                std::cerr << util::pose_mat2vec(camera_poses[obs3.first]);
                std::cerr << std::endl;
                */
                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<triangulation3D, 3, 3>(
                            new triangulation3D(
                                obs3.second.x,
                                obs3.second.y,
                                obs3.second.z,
                                camera_poses[obs3.first][0],
                                camera_poses[obs3.first][1],
                                camera_poses[obs3.first][2],
                                camera_poses[obs3.first][3],
                                camera_poses[obs3.first][4],
                                camera_poses[obs3.first][5]
                                )
                            );
                problem.AddResidualBlock(
                        cost_function,
                        new ceres::TrivialLoss,
                        //new ceres::CauchyLoss(loss_thresh_3D3D),
                        transform);
                if(!initialized) {
                    initialized = 3;
                    ceres::Solve(options, &problem, &summary);
                }
            }
        }
        for(int cam=0; cam<num_cams; cam++) {
            for(auto obs2 : solver_frames->keypoint_obs2[id][cam]) {
                /*
                std::cerr << "2D observation " <<  obs2.second << ": ";
                for(int i=0; i<6; i++) {
                    std::cerr << camera_poses[obs2.first][i] << " ";
                }
                std::cerr << ", " << cam_trans[cam].transpose();
                std::cerr << std::endl;
                */
                ceres::CostFunction* cost_function =
                    new ceres::AutoDiffCostFunction<triangulation2D, 2, 3>(
                            new triangulation2D(
                                obs2.second.x,
                                obs2.second.y,
                                camera_poses[obs2.first][0],
                                camera_poses[obs2.first][1],
                                camera_poses[obs2.first][2],
                                camera_poses[obs2.first][3],
                                camera_poses[obs2.first][4],
                                camera_poses[obs2.first][5],
                                solver_track->cam_trans[cam](0),
                                solver_track->cam_trans[cam](1),
                                solver_track->cam_trans[cam](2)
                                )
                            );
                problem.AddResidualBlock(
                        cost_function,
                        new ceres::ScaledLoss(
                            new ceres::CauchyLoss(loss_thresh_3D2D),
                            weight_3D2D,
                            ceres::TAKE_OWNERSHIP),
                        transform);
            }
        }
        ceres::Solve(options, &problem, &summary);
        point.x = transform[0];
        point.y = transform[1];
        point.z = transform[2];

    }
    void Solver::output_line(Eigen::Matrix4d result, std::ofstream &output)
    {
        output<< result(0,0) << " "
            << result(0,1) << " "
            << result(0,2) << " "
            << result(0,3) << " "
            << result(1,0) << " "
            << result(1,1) << " "
            << result(1,2) << " "
            << result(1,3) << " "
            << result(2,0) << " "
            << result(2,1) << " "
            << result(2,2) << " "
            << result(2,3) << " "
            << std::endl;
    }
    void Solver::iSAM_print_stats(const int frame)
    {
        calib_isam->update();
        calib_isam->print_stats();
        if(frame > 0 && frame % ba_every == 0 || frame == num_frames-1) {
            calib_isam->update();
            if(frame == num_frames-1) {
                calib_isam->batch_optimization();
            }

            std::ofstream output;
            output.open(("results/" + file_name + ".txt").c_str());
            for(int i=0; i<=frame; i++) {
                auto node = cam_nodes[0][i];
                Eigen::Matrix4d result = node->value().wTo();
                output_line(result, output);
            }
            output.close();
        }


    }



}
