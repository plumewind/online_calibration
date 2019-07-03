#include "System.h"

namespace online_calibration
{
	System::System(int num_cams_, int num_frames_, std::string file_name_)
		:num_cams(num_cams_), num_frames(num_frames_), file_name(file_name_)
	{
		ndiagonal = 4;
		min_matches = 0;
		id_counter = 0;
		detect_every = 1;

		loop_close_thresh = 10, // meters
		agreement_t_thresh = 0.1, // meters
		agreement_r_thresh = 0.05; // radians

	}
	System::~System()
	{
	
	}
	void System::system_init(std::vector<Eigen::Matrix<float, 3, 4>,
    						Eigen::aligned_allocator<Eigen::Matrix<float, 3, 4>>>& cam_mat,
						Eigen::Matrix4f& velo2cam)
	{
		// attempt matching frames between dframes
		dframes = std::vector<std::vector<int>>(2);

		//ifdef ENABLE_ISAM
		std::vector<int> dframe_;
		for(int k=1; k<=ndiagonal; k++) {
        	dframe_.push_back(k);
    	}
		dframes.push_back(dframe_);
		img_prevs = std::vector<cv::Mat>(num_cams);

		sys_Frames = new CameraFrame(num_cams, num_frames);
		sys_Tracking = new Tracking(num_cams, sys_Frames);
		sys_Tracking->setCameraKinv(velo2cam, cam_mat);

		sys_Solver = new Solver(file_name, num_frames, num_cams, sys_Frames, sys_Tracking);
		sys_Viewer = new Viewer(num_cams, sys_Frames, sys_Tracking);

	}
	void System::run(const cv::Mat &left_img, 
				const cv::Mat &right_img, 
				const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans, 
				int frame
				)
	{
		//sys_Tracking->setCameraKinv(velo2cam, cam_mat);
		
		std::vector<cv::Mat> raw_images{left_img, right_img};

		sys_Solver->isam_add_node(frame);
		if(frame > 0) {//从第二帧开始处理
            for(int cam = 0; cam<num_cams; cam++) {//枚举摄像头
                for(int prev_cam = 0; prev_cam < num_cams; prev_cam++) {//再次枚举摄像头
                    //左目k-1和左目k，右目k-1和左目k，左目k-1和右目k，右目k-1和右目k
                    sys_Tracking->trackFeatures(raw_images[prev_cam], raw_images[cam], prev_cam, cam, frame-1, frame);
				}
			}
			for(int cam = 0; cam<num_cams; cam++) {//枚举摄像头
                sys_Tracking->consolidateFeatures(cam, frame);
				sys_Tracking->removeTerribleFeatures(raw_images[cam], cam, frame);

				std::vector<std::vector<cv::Point2f>> projection;
                std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scans_valid;
                sys_Tracking->projectLidarToCamera(scans, projection, scans_valid, cam);//将激光点投影在图像平面

				sys_Frames->kp_with_depth[cam][frame] =
                    pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
				sys_Tracking->featureDepthAssociation(scans_valid, projection, cam, frame);

				sys_Viewer->view_depth_assoc(raw_images[cam], cam, frame, projection, scans_valid);
			}
		}
		std::cerr << "System::run Feature tracking done" << std::endl;

		//优化解相机位姿
		double transform[6] = {0, 0, 0, 0, 0, 1};
		for(int ba = 0; ba < 2; ba++) {
			//std::cout << "System::run  test0 dframes.size = "<<(dframes.size())<<std::endl;
            for(int dframe : dframes[ba]) {
				//std::cout << "System::run  test1"<<std::endl;
				if(sys_Solver->pose_init(ba, frame, dframe, transform)==false)
					break;
				//std::cout << "System::run  test2"<<std::endl;
				// if attempting loop closure,
                // check if things are close by
                if(sys_Solver->loopclosure_compute(ba, transform)==false)
					continue;
                std::cerr << "Computing f2f pose: "<< " " << frame-dframe << "-" << frame
                    		<< " ba=" << ba << std::endl;

				std::vector<std::vector<std::pair<int, int>>> matches(num_cams);
                std::vector<std::vector<std::pair<int, int>>> good_matches(num_cams);
                std::vector<std::vector<ResidualType>> residual_type(num_cams);
				if(ba == 0) {
                    sys_Solver->matchUsingId(sys_Frames->keypoint_ids, frame, frame-dframe, matches);
                    std::cerr << "Matches using id: ";
                    for(int zxcv=0; zxcv<num_cams; zxcv++) {
                        std::cerr << matches[zxcv].size() << " ";
                    }
                    std::cerr << std::endl;
                } else {
                    sys_Solver->matchFeatures(sys_Frames->descriptors, frame, frame-dframe, matches);
                    std::cerr << "Matches using descriptors: ";
                    for(int zxcv=0; zxcv<num_cams; zxcv++) {
                        std::cerr << matches[zxcv].size() << " ";
                    }
                    std::cerr << std::endl;
                }
				if(dframe > 1 && matches[0].size() < min_matches) 
					break;
                
				//根据匹配数量确定是否需要使用ICP
				bool enable_icp = ba;
                if(matches[0].size() < 100) {
                    if(ba == 0 && dframe != 1) break;
                    enable_icp = true;
                }

				//输出预测的位姿
                std::cerr << "Predicted: ";
                for(int i=0; i<6; i++) std::cerr << transform[i] << " ";
                std::cerr << std::endl;

				// get triangulated landmarks
                std::map<int, pcl::PointXYZ> landmarks_at_frame;
                std::cerr << "Getting triangulated landmarks at frame "
                    << frame-dframe << std::endl;
				sys_Solver->getLandmarksAtFrame(frame, dframe, landmarks_at_frame);

				//帧间运动估计
				auto start = clock() / double(CLOCKS_PER_SEC);
				Eigen::Matrix4d dpose = sys_Solver->frameToFrame(matches, landmarks_at_frame,
											frame, frame-dframe, transform,
											good_matches, residual_type, enable_icp);
				auto end = clock() / double(CLOCKS_PER_SEC);

				if(dframe == 1) {
                    sys_Solver->ceres_poses_mat[frame] = sys_Solver->ceres_poses_mat[frame-1] * dpose;
                    Tools::pose_vec2mat(sys_Solver->ceres_poses_mat[frame], sys_Solver->ceres_poses_vec[frame]);
                }
                std::cerr << "Optimized (t=" << end - start << "): ";
                for(int i=0; i<6; i++) std::cerr << transform[i] << " ";
                std::cerr << std::endl;

				// check agreement
                double agreement[6];
				Eigen::Matrix4d dT = Tools::pose_mat2vec(transform);
                Tools::pose_vec2mat(dpose * dT.inverse(), agreement);
                std::cerr << "Agreement: ";
                for(int i=0; i<6; i++) std::cerr << agreement[i] << " ";
                std::cerr << std::endl;
                Eigen::Vector3d agreement_t;
                agreement_t << agreement[3], agreement[4], agreement[5];
                Eigen::Vector3d agreement_r;
                agreement_r << agreement[0], agreement[1], agreement[2];

                if(ba == 0 && agreement_t.norm() >
                        std::min(agreement_t_thresh * dframe, loop_close_thresh)
                        && dframe > 1) {
                    std::cerr << "Poor t agreement: " << agreement_t.norm()
                        << " " << agreement_t_thresh << " " << dframe
                        << std::endl;
                    continue;
                }
                if(agreement_r.norm() > agreement_r_thresh && dframe > 1) {
                    std::cerr << "Poor r agreement" << std::endl;
                    continue;
                }
                if(ba == 1) {
                    std::cerr << "Loop closed!" << std::endl;
                }

				sys_Viewer->view_features(frame, dframe, raw_images,
										matches, good_matches, residual_type);

				if(dframe == 1) 
                    sys_Tracking->removeSlightlyLessTerribleFeatures(frame, good_matches);

				sys_Solver->iSAM_update(frame, dframe, dpose);

			}
		}

		// Detect new features
		std::cerr << "System::run Detect new features !" << std::endl;
		for(int cam = 0; cam<num_cams; cam++) {
            if(frame % detect_every == 0) {
				//std::cout << "System::run  test1"<<std::endl;
                sys_Tracking->detectFeatures(raw_images[cam], id_counter, cam, frame);
				if(cam == 0) {
                    for(int other_cam = 0; other_cam < num_cams; other_cam++) {
                        if(other_cam == cam) continue;
                        sys_Tracking->trackFeatures(raw_images[cam], raw_images[other_cam],
													cam, other_cam, frame, frame );
					}
				}
			}
		}

		std::cout << "System::run  test2"<<std::endl;
		for(int cam = 0; cam<num_cams; cam++) {
            //std::cerr << "inter frame tracked" << std::endl;
            // TODO: don't do this twice
            sys_Tracking->consolidateFeatures(cam, frame);
			sys_Tracking->removeTerribleFeatures(raw_images[cam], cam, frame);

			std::vector<std::vector<cv::Point2f>> projection;
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scans_valid;
			sys_Tracking->projectLidarToCamera(scans, projection, scans_valid, cam);

			sys_Frames->kp_with_depth[cam][frame].reset();
            sys_Frames->kp_with_depth[cam][frame] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
            sys_Tracking->featureDepthAssociation(scans_valid, projection, cam, frame);
			img_prevs[cam] = raw_images[cam];
		}
		std::cout << "System::run  test3"<<std::endl;
		sys_Solver->iSAM_add_keypoint(frame, id_counter, true);
		std::cout << "System::run  test3.5 frame = "<<(frame)<<std::endl;
		sys_Solver->iSAM_add_measurement(frame);
		std::cout << "System::run  test4"<<std::endl;
		sys_Solver->iSAM_print_stats(frame);

		std::cerr << "System::run Frame complete: " << frame << std::endl;
	}
	void System::run(const cv::Mat &img, 
				const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans, 
				int frame)
	{



	}








}
