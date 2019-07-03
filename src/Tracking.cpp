#include "Tracking.h"

namespace online_calibration
{
	Tracking::Tracking(int num_cams_, CameraFrame* track_frames_mp)
		:track_frames(track_frames_mp),num_cams(num_cams_)
	{

		flow_outlier = 20000.0; // pixels^2, squared distance of optical flow
		quality_level = 0.001; // good features to track quality
		min_distance = 12;// pixel distance between nearest features
		lkt_window = 21;
		lkt_pyramid = 4; 
		corner_count = 3000; // number of features
		match_thresh = 29;
		depth_assoc_thresh = 0.015;

		img_width = 1226;
		img_height = 370; // kitti 数据集中图像的宽度和高度
		 // FREAK feature descriptor
		track_freak = cv::xfeatures2d::FREAK::create(
				false, // orientation normalization
				false // scale normalization
				);

		// good features to detect
		track_gftt = cv::GFTTDetector::create(
				corner_count,
				quality_level,
				min_distance);
	}
	Tracking::~Tracking()
	{
	
	}
	void Tracking::setCameraKinv(
				Eigen::Matrix4f& velo2cam,
				std::vector<Eigen::Matrix<float, 3, 4>,
    			Eigen::aligned_allocator<Eigen::Matrix<float, 3, 4>>>& cam_mat)
	{
		for(int cam=0; cam<cam_mat.size(); cam++) {

			Eigen::Matrix3f K = cam_mat[cam].block<3,3>(0,0);
			Eigen::Matrix3f Kinv = K.inverse();
			Eigen::Vector3f Kt = cam_mat[cam].block<3,1>(0,3);
			Eigen::Vector3f t = Kinv * Kt;
			cam_trans.push_back(t);
			cam_intrinsic.push_back(K);
			cam_intrinsic_inv.push_back(K.inverse());

			cam_pose.push_back(Eigen::Matrix4f::Identity());
			cam_pose[cam](0, 3) = t(0);
			cam_pose[cam](1, 3) = t(1);
			cam_pose[cam](2, 3) = t(2);

			Eigen::Vector3f min_pt;
			min_pt << 0, 0, 1;
			min_pt = Kinv * min_pt;
			min_x.push_back(min_pt(0) / min_pt(2));
			min_y.push_back(min_pt(1) / min_pt(2));
			//std::cerr << "min_pt: " << min_pt << std::endl;
			Eigen::Vector3f max_pt;
			max_pt << img_width, img_height, 1;
			max_pt = Kinv * max_pt;
			max_x.push_back(max_pt(0) / max_pt(2));
			max_y.push_back(max_pt(1) / max_pt(2));
			//std::cerr << "max_pt: " << max_pt << std::endl;
		}
		velo_to_cam = velo2cam;
		cam_to_velo = velo_to_cam.inverse();
	}
	void Tracking::trackFeatures(
        // std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints,
        // std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints_p,
        // std::vector<std::vector<std::vector<int>>> &keypoint_ids,
        // std::vector<std::vector<cv::Mat>> &descriptors,
        const cv::Mat &img1,
        const cv::Mat &img2,
        const int cam1,
        const int cam2,
        const int frame1,
        const int frame2
        ) {
		static bool init_flag = false;
		if(init_flag == false){
			img_width = img1.cols;
			img_height = img1.rows;
			init_flag = true;
		}
		const Eigen::Matrix3f &Kinv1 = cam_intrinsic_inv[cam1];
		const Eigen::Matrix3f &Kinv2 = cam_intrinsic_inv[cam2];

		int m = track_frames->keypoints[cam1][frame1].size();         //统计上一帧相机归一化坐标的数量
		if(m == 0) {
			std::cerr << "ERROR: No features to track." << std::endl;
		}
		std::vector<cv::Point2f> points1(m), points2(m);
		for(int i=0; i<m; i++) {
			points1[i] = track_frames->keypoints_p[cam1][frame1][i];  //提取上一帧相机像素坐标
		}
		std::vector<unsigned char> status;              //存储TK光流匹配后的状态
		std::vector<float> err;
		cv::calcOpticalFlowPyrLK(
				img1,
				img2,
				points1,
				points2,
				status,
				err,
				cv::Size(lkt_window, lkt_window),
				lkt_pyramid,
				cv::TermCriteria(
					CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
					30,
					0.01
					),
				0
				);
		// int col_cells = img_width / min_distance + 2,
		//     row_cells = img_height / min_distance + 2;
		//std::vector<std::vector<cv::Point2f>> occupied(col_cells * row_cells);
		for(int i=0; i<m; i++) {
			if(!status[i]) {
				continue;
			}
			if(Tools::dist2(points1[i], points2[i]) > flow_outlier) {
				continue;
			}
			// somehow points can be tracked to negative x and y
			if(points2[i].x < 0 || points2[i].y < 0 ||
					points2[i].x >= img_width ||
					points2[i].y >= img_height) {
				continue;
			}

			track_frames->keypoints_p[cam2][frame2].push_back(points2[i]);//保存已跟踪点的相机像素坐标
			track_frames->keypoints[cam2][frame2].push_back(              //保存已跟踪点的相机归一化坐标
					CameraFrame::pixel2canonical(points2[i], Kinv2)
					);
			track_frames->keypoint_ids[cam2][frame2].push_back(           //保存已跟踪点的id
					track_frames->keypoint_ids[cam1][frame1][i]);
			track_frames->descriptors[cam2][frame2].push_back(
					track_frames->descriptors[cam1][frame1].row(i).clone());
		}
	}
	void Tracking::consolidateFeatures(
	const int cam,
	const int frame
	){
		std::vector<cv::Point2f> *keypoints = &(track_frames->keypoints[cam][frame]);
        std::vector<cv::Point2f> *keypoints_p = &(track_frames->keypoints_p[cam][frame]);
        std::vector<int> *keypoint_ids = &(track_frames->keypoint_ids[cam][frame]);
        cv::Mat *descriptors = &(track_frames->descriptors[cam][frame]);

		// merges keypoints of the same id using the geometric median
		// geometric median is computed in canonical coordinates
		const Eigen::Matrix3f &K = cam_intrinsic[cam];
		int m = keypoint_ids->size();
		std::map<int, std::vector<int>> keypoints_map;
		for(int i=0; i<m; i++) {
			keypoints_map[(*keypoint_ids)[i]].push_back(i);
		}
		int mm = keypoints_map.size();
		std::vector<cv::Point2f> tmp_keypoints(mm);
		std::vector<cv::Point2f> tmp_keypoints_p(mm);
		std::vector<int> tmp_keypoint_ids(mm);
		cv::Mat tmp_descriptors(mm, descriptors->cols, descriptors->type());
		int mi = 0;
		for(auto kp : keypoints_map) {
			int id = kp.first;
			int n = kp.second.size();

			cv::Point2f gm_keypoint;
			if(n > 2) {
				std::vector<cv::Point2f> tmp_tmp_keypoints(n);
				for(int i=0; i<n; i++) {
					int j = kp.second[i];
					tmp_tmp_keypoints[i] = (*keypoints)[j];
				}
				gm_keypoint = Tools::geomedian(tmp_tmp_keypoints);
			} else if(n ==2) {
				gm_keypoint = (
						(*keypoints)[kp.second[0]] + 
						(*keypoints)[kp.second[1]])/2;
			} else {
				gm_keypoint = (*keypoints)[kp.second[0]];
			}
			tmp_keypoint_ids[mi] = id;
			tmp_keypoints[mi] = gm_keypoint;
			tmp_keypoints_p[mi] = CameraFrame::canonical2pixel(gm_keypoint, K);
			descriptors->row(kp.second[0]).copyTo(tmp_descriptors.row(mi));
			mi++;
		}

		*keypoints = tmp_keypoints;
		*keypoints_p = tmp_keypoints_p;
		*keypoint_ids = tmp_keypoint_ids;
		tmp_descriptors.copyTo(*descriptors);
	}
	void Tracking::removeTerribleFeatures(
		//std::vector<cv::Point2f> &keypoints,
		//std::vector<cv::Point2f> &keypoints_p,
		//std::vector<int> &keypoint_ids,
		//cv::Mat &descriptors,
		//const cv::Ptr<cv::DescriptorExtractor> extractor,
		const cv::Mat &img,
		const int cam,
		const int frame
	)
	{
		std::vector<cv::Point2f> *keypoints = &(track_frames->keypoints[cam][frame]);
        std::vector<cv::Point2f> *keypoints_p = &(track_frames->keypoints_p[cam][frame]);
        std::vector<int> *keypoint_ids = &(track_frames->keypoint_ids[cam][frame]);
        cv::Mat *descriptors = &(track_frames->descriptors[cam][frame]);

		// remove features if the extracted descriptor doesn't match
		std::vector<cv::KeyPoint> cvKP(keypoints_p->size());
		for(int i=0; i<keypoints_p->size(); i++) {
			cvKP[i].pt = (*keypoints_p)[i];
		}
		std::vector<cv::Point2f> tmp_keypoints;
		std::vector<cv::Point2f> tmp_keypoints_p;
		std::vector<int> tmp_keypoint_ids;
		cv::Mat tmp_descriptors, tmp_tmp_descriptors;

		int i=0;
		track_freak->compute(img, cvKP, tmp_tmp_descriptors);
		for(int j=0; j<cvKP.size(); j++) {
			while(cv::norm((*keypoints_p)[i] - cvKP[j].pt) > kp_EPS) {
				i++;
			}
			if(cv::norm((*descriptors).row(i),
						tmp_tmp_descriptors.row(j),
						cv::NORM_HAMMING) < match_thresh) {
				tmp_keypoints.push_back((*keypoints)[i]);
				tmp_keypoints_p.push_back((*keypoints_p)[i]);
				tmp_keypoint_ids.push_back((*keypoint_ids)[i]);
				tmp_descriptors.push_back((*descriptors).row(i).clone());
			}
		}
		*keypoints = tmp_keypoints;
		*keypoints_p = tmp_keypoints_p;
		*keypoint_ids = tmp_keypoint_ids;
		tmp_descriptors.copyTo(*descriptors);
	}
	void Tracking::projectLidarToCamera(
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans,
        std::vector<std::vector<cv::Point2f>> &projection,
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans_valid,
        const int cam
        )
	{
		int bad = 0;
		Eigen::Vector3f t = cam_trans[cam];

		for(int s=0; s<scans.size(); s++) {
			pcl::PointCloud<pcl::PointXYZ> projected_points;
			projection.push_back(std::vector<cv::Point2f>());
			scans_valid.push_back(pcl::PointCloud<pcl::PointXYZ>::Ptr(
						new pcl::PointCloud<pcl::PointXYZ>));
			for(int i=0, _i = scans[s]->size(); i<_i; i++) {//枚举点云中每一个激光点
				pcl::PointXYZ p = scans[s]->at(i);                      //从点云中取出每一个激光点
				pcl::PointXYZ pp(p.x + t(0), p.y + t(1), p.z + t(2));   //将激光点（已在相机坐标系中）投影到相机成像平面
				cv::Point2f c(pp.x/pp.z, pp.y/pp.z);                    //归一化成相机像素坐标
				if(pp.z > 0 && c.x >= min_x[cam] && c.x < max_x[cam]
						&& c.y >= min_y[cam] && c.y < max_y[cam]) {
					// remove points occluded by current point
					while(projection[s].size() > 0
							&& c.x < projection[s].back().x
							&& pp.z < projected_points.back().z) {//栈内点在当前点的右边且深度大于当前点，则栈內点出栈
						projection[s].pop_back();
						projected_points.points.pop_back();
						scans_valid[s]->points.pop_back();
						bad++;
					}
					// ignore occluded points
					if(projection[s].size() > 0
							&& c.x < projection[s].back().x        
							&& pp.z > projected_points.back().z) {//栈内点在当前点的右边且深度小于当前点，则当前点忽略
						bad++;
						continue;
					}
					projection[s].push_back(c);
					projected_points.push_back(pp);                 //保存合格激光点的临时堆栈
					scans_valid[s]->push_back(scans[s]->at(i));     //保存合格激光点的源坐标
				}
			}
			// std::cerr << s << " " << scans_valid[s]->size()
			//     << " " << scans_valid[s]->points.size() << std::endl;
		}
		// std::cerr << "Lidar projection bad: " << bad << std::endl;
	}
	void Tracking::featureDepthAssociation(
		const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans,//合格激光点云
		const std::vector<std::vector<cv::Point2f>> &projection,      //合格激光点的投影点
		//const std::vector<cv::Point2f> &keypoints,                    //图像特征点
		//pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_with_depth,
		//std::vector<int> &has_depth
		const int cam,
		const int frame
        )
	{
		std::vector<cv::Point2f> *keypoints = &(track_frames->keypoints[cam][frame]);
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_with_depth = (track_frames->kp_with_depth[cam][frame]);
		std::vector<int> *has_depth = &(track_frames->has_depth[cam][frame]);

		has_depth->resize(keypoints->size());
		for(int i=0; i<has_depth->size(); i++) {//将深度标志数组置初始状态
			(*has_depth)[i] = -1;
		}
		/*
		std::cerr << "Sizes: " <<  scans.size() << " "
			<< projection.size() << " "
			<< keypoints.size() << std::endl;
		*/
		int has_depth_n = 0;
		for(int k=0; k<keypoints->size(); k++) {//枚举每一个特征点
			(*has_depth)[k] = -1;                 //默认特征点没有深度值
			cv::Point2f kp = (*keypoints)[k];
			int last_interp = -1;
			for(int s=0, _s = scans.size(); s<_s; s++) {//对于每一个合格激光点云
				bool found = false;
				if(projection[s].size() <= 1) {
					last_interp = -1;
					continue;
				}
				int lo = 0, hi = projection[s].size() - 2, mid = 0;
				while(lo <= hi) {//在x轴上，利用二分查找最近点
					mid = (lo + hi)/2;
					if(projection[s][mid].x > kp.x) {
						hi = mid-1;
					} else if(projection[s][mid+1].x <= kp.x) {
						lo = mid+1;
					} else {
						found = true;
						if(last_interp != -1
								&& (projection[s][mid].y > kp.y) !=
									(projection[s-1][last_interp].y > kp.y)//保证待求点在上下两个点之间
								&& abs(projection[s][mid].x -              //保证第s层两个插值点有一定间距
									projection[s][mid+1].x)
									< depth_assoc_thresh
								&& abs(projection[s-1][last_interp].x -    //保证第s-1层两个插值点有一定间距
									projection[s-1][last_interp+1].x)
									< depth_assoc_thresh
								) {
							/*
							* Perform linear interpolation using four points:
							* 第s-1层点云, 第last_interp个点 ----- interp2 ----- 第s-1层点云, 第last_interp+1个点
							*                                      |
							*                                      kp
							*                                      |
							*      第s层点云, 第mid个点 ---------- interp1 --------- 第s层点云, 第mid+1个点
							*/
							/*
							std::cerr << "depth association: " << kp
								<< " " << projection[s][mid]
								<< " " << projection[s][mid+1]
								<< " " << projection[s-1][last_interp]
								<< " " << projection[s-1][last_interp+1]
								<< std::endl;
							std::cerr << "                   "
								<< " " << scans[s]->at(mid)
								<< " " << scans[s]->at(mid+1)
								<< " " << scans[s-1]->at(last_interp)
								<< " " << scans[s-1]->at(last_interp+1)
								<< " ";
								*/
							pcl::PointXYZ interp1 = Tools::linterpolate(//在第s层点云上进行x轴方向上的插值
									scans[s]->at(mid),
									scans[s]->at(mid+1),
									projection[s][mid].x,
									projection[s][mid+1].x,
									kp.x);
							pcl::PointXYZ interp2 = Tools::linterpolate(//在第s-1层点云上进行x轴方向上的插值
									scans[s-1]->at(last_interp),
									scans[s-1]->at(last_interp+1),
									projection[s-1][last_interp].x,
									projection[s-1][last_interp+1].x,
									kp.x);
							float i1y = Tools::linterpolate(//第s层点云投影之后的平面点y方向上存在偏差，通过x轴进行y轴插值，减小误差
									projection[s][mid].y,
									projection[s][mid+1].y,
									projection[s][mid].x,
									projection[s][mid+1].x,
									kp.x);
							float i2y = Tools::linterpolate(//第s-1层点云投影之后也是如此
									projection[s-1][last_interp].y,
									projection[s-1][last_interp+1].y,
									projection[s-1][last_interp].x,
									projection[s-1][last_interp+1].x,
									kp.x);

							pcl::PointXYZ kpwd = Tools::linterpolate(//在第“kp”层（待求点所在的层）点云上进行y轴方向上的插值
									interp1,
									interp2,
									i1y,
									i2y,
									kp.y);

							// std::cerr <<"kpwd : "<< kpwd << std::endl;
							// std::cerr <<"kp : "<< kp << std::endl;
							// exit(1);

							keypoints_with_depth->push_back(kpwd);//保存带深度信息的图像特征点
							(*has_depth)[k] = has_depth_n;             //记录已经插值的个数，即插值点序号
							has_depth_n++;                          //插值序号自加
						}
						last_interp = mid;//保存当前插值点
						break;
					}
				}
				if(!found) {
					last_interp = -1;
				}
				if((*has_depth)[k] != -1) break;
			}
		}
		/*
		std::cerr << "Has depth: " << has_depth_n << "/" << keypoints.size() << std::endl;
		*/
		//return has_depth;
	}

	void Tracking::detectFeatures(
        // std::vector<std::vector<cv::Point2f>> &keypoints,
        // std::vector<std::vector<cv::Point2f>> &keypoints_p,
        // std::vector<std::vector<int>> &keypoint_ids,
        // std::vector<cv::Mat> &descriptors,
        // const cv::Ptr<cv::FeatureDetector> detector,
        // const cv::Ptr<cv::DescriptorExtractor> extractor,
        const cv::Mat &img,
        int &id_counter,
        const int cam,
        const int frame
        ) {
		//std::cout << "Tracking::detectFeatures  test1 cam_intrinsic_inv.size = "<<(cam_intrinsic_inv.size())<<std::endl;
		const Eigen::Matrix3f &Kinv = cam_intrinsic_inv[cam];

		int col_cells = img_width / min_distance + 2,
			row_cells = img_height / min_distance + 2;
		std::vector<std::vector<cv::Point2f>> occupied(col_cells * row_cells);
		for(cv::Point2f p : track_frames->keypoints_p[cam][frame]) {
			int col = p.x / min_distance,
				row = p.y / min_distance;
			occupied[col * row_cells + row].push_back(p);
		}

		std::vector<cv::KeyPoint> cvKP;
		track_gftt->detect(img, cvKP);
		cv::Mat tmp_descriptors;
		// remember! compute MUTATES cvKP
		track_freak->compute(img, cvKP, tmp_descriptors);
		int detected = 0;
		for(int kp_i = 0; kp_i < cvKP.size(); kp_i ++) {
			auto kp = cvKP[kp_i];
			//std::cerr << kp.size << " " << kp.angle << " " << kp.response << std::endl;
			int col = kp.pt.x / min_distance,
				row = kp.pt.y / min_distance;
			bool bad = false;
			int col_start = std::max(col-1, 0),
				col_end = std::min(col+1, col_cells-1),
				row_start = std::max(row-1, 0),
				row_end = std::min(row+1, row_cells-1);
			float md2 = min_distance * min_distance;
			for(int c=col_start; c<=col_end && !bad; c++) {
				for(int r=row_start; r<=row_end && !bad; r++) {
					for(auto pp : occupied[c * row_cells + r]) {
						if(Tools::dist2(pp, kp.pt) < md2) {
							bad = true;
							break;
						}
					}
				}
			}
			if(bad) continue;
			track_frames->keypoints_p[cam][frame].push_back(kp.pt);
			track_frames->keypoints[cam][frame].push_back(
					CameraFrame::pixel2canonical(kp.pt, Kinv)
					);
			track_frames->descriptors[cam][frame].push_back(tmp_descriptors.row(kp_i).clone());
			track_frames->keypoint_ids[cam][frame].push_back(id_counter++);
			detected++;
		}
		//std::cerr << "Detected: " << detected << std::endl;
	}
	void Tracking::removeSlightlyLessTerribleFeatures(
        //std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints,
        //std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints_p,
        //std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> &kp_with_depth,
        //std::vector<std::vector<std::vector<int>>> &keypoint_ids,
        //std::vector<std::vector<cv::Mat>> &descriptors,
        //std::vector<std::vector<std::vector<int>>> &has_depth,
        const int frame,
        const std::vector<std::vector<std::pair<int, int>>> &good_matches)
	{
		// std::vector<cv::Point2f> *keypoints = &(track_frames->keypoints[cam][frame]);
		// std::vector<cv::Point2f> *keypoints_p = &(track_frames->keypoints_p[cam][frame]);
		// std::vector<int> *keypoint_ids = &(track_frames->keypoint_ids[cam][frame]);
		// cv::Mat *descriptors = &(track_frames->descriptors[cam][frame]);

		// remove features not matched in good_matches
		for(int cam=0; cam<num_cams; cam++) {
			// I'm not smart enough to figure how to delete things
			// other than making a new vector, pointcloud, or mat
			// and then copying everything over :(
			// In Python I could have just written a = a[indices]
			std::set<int> good_indices;
			for(auto gm : good_matches[cam]) {
				good_indices.insert(gm.first);
			}
			int m = good_indices.size(), n = track_frames->keypoints[cam][frame].size();
			std::cerr << "Good matches of " << cam << ": " << m << "/" << n << std::endl;
			std::vector<cv::Point2f> tmp_keypoints(m);
			std::vector<cv::Point2f> tmp_keypoints_p(m);
			pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_kp_with_depth(
					new pcl::PointCloud<pcl::PointXYZ>);
			std::vector<int> tmp_keypoint_ids(m);
			cv::Mat tmp_descriptors(
					m, track_frames->descriptors[cam][frame].cols,
					track_frames->descriptors[cam][frame].type());
			std::vector<int> tmp_has_depth(m);
			int j = 0, jd = 0;
			for(int i=0; i<n; i++) {
				if(!good_indices.count(i)) continue;
				tmp_keypoints[j] = track_frames->keypoints[cam][frame][i];
				tmp_keypoints_p[j] = track_frames->keypoints_p[cam][frame][i];
				tmp_keypoint_ids[j] = track_frames->keypoint_ids[cam][frame][i];
				track_frames->descriptors[cam][frame].row(i).copyTo(tmp_descriptors.row(j));
				int d = track_frames->has_depth[cam][frame][i];
				if(d != -1) {
					tmp_kp_with_depth->push_back(track_frames->kp_with_depth[cam][frame]->at(d));
					tmp_has_depth[j] = jd++;
				} else {
					tmp_has_depth[j] = -1;
				}
				j++;
			}
			// these are all allocated on the stack or smart pointers
			// so no memory leaks, hopefully
			track_frames->keypoints[cam][frame] = tmp_keypoints;
			track_frames->keypoints_p[cam][frame] = tmp_keypoints_p;
			track_frames->kp_with_depth[cam][frame] = tmp_kp_with_depth;
			track_frames->keypoint_ids[cam][frame] = tmp_keypoint_ids;
			tmp_descriptors.copyTo(track_frames->descriptors[cam][frame]);
			track_frames->has_depth[cam][frame] = tmp_has_depth;
    }








	}

}
