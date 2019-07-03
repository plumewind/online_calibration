#include "Viewer.h"

namespace online_calibration
{
	Viewer::Viewer(int num_cams_, CameraFrame* view_frames_mp, Tracking* view_track_mp)
		:view_frames(view_frames_mp), view_track(view_track_mp),num_cams(num_cams_)
	{

		features_windown = "features_detect";
		cvNamedWindow(features_windown);
		cv::startWindowThread();

		depthassoc_windown = "depth_feature_assoc";
		cvNamedWindow(depthassoc_windown);
		cv::startWindowThread();

	}
	Viewer::~Viewer()
	{
	
	}
	void Viewer::view_depth_assoc(cv::Mat &imgs, int cam, int  frame,
			std::vector<std::vector<cv::Point2f>>& projection,
            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& scans_valid)
	{
		if(cam == 0) {
			cv::Mat draw;
			cvtColor(imgs, draw, cv::COLOR_GRAY2BGR);  //转化成灰度信息显示
			auto &K = view_track->cam_intrinsic[cam];
			for(int s=0, _s = projection.size(); s<_s; s++) {           //枚举每一个激光投影点云
				auto P = projection[s];
				for(int ss=0; ss<projection[s].size(); ss++) {          //枚举每一个激光点
					auto pp = CameraFrame::canonical2pixel(projection[s][ss], K);    //归一化坐标转化成像素坐标
					auto PP = scans_valid[s]->at(ss);                   //提取激光点源坐标
					int D = 200;
					double d = sqrt(PP.z * 5/D) * D;                    //按照激光点深度进行不同颜色深浅的绘制
					if(d > D) d = D;
					cv::circle(draw, pp, 1, cv::Scalar(0, D-d, d), -1, 8, 0);           //颜色scalar--BGRA
				}
			}
			for(int k=0; k<view_frames->keypoints[cam][frame].size(); k++) {//枚举每一个特征点
				auto p = view_frames->keypoints_p[cam][frame][k];//提取特征点的像素坐标
				int hd = view_frames->has_depth[cam][frame][k];  //提出特征点的插值id号
				if(hd != -1) {//如果进行过插值
					int D = 255;
					double d = sqrt(
							view_frames->kp_with_depth[cam][frame]->at(hd).z * 5/D) * D;//同样计算深度信息
					if(d > D) d = D;
					cv::circle(draw, p, 4, cv::Scalar(0, 255-d, d), -1, 8, 0);
					cv::circle(draw, p, 4, cv::Scalar(0, 0, 0), 1, 8, 0);//黑色
				} else {
					cv::circle(draw, p, 4, cv::Scalar(255, 200, 0), -1, 8, 0);//天蓝色
					cv::circle(draw, p, 4, cv::Scalar(0, 0, 0), 1, 8, 0);
				}
			}
			imshow(depthassoc_windown, draw);
		}
	}
	void Viewer::view_features(int frame, int dframe,std::vector<cv::Mat>& imgs,
							std::vector<std::vector<std::pair<int, int>>>& matches,
							std::vector<std::vector<std::pair<int, int>>>& good_matches,
							std::vector<std::vector<ResidualType>>& residual_type)
	{
		if(dframe == 1) {
			std::vector<cv::Mat> draws(num_cams);
			for(int cam=0; cam<num_cams; cam++) {
				cv::Mat draw;
				cvtColor(imgs[cam], draw, cv::COLOR_GRAY2BGR);
				auto &K = view_track->cam_intrinsic[cam];
				//cv::drawKeypoints(img, keypoints[frame], draw);
				for(int k=0; k<view_frames->keypoints[cam][frame].size(); k++) {
					auto p = view_frames->keypoints_p[cam][frame][k];
					if(view_frames->has_depth[cam][frame][k] != -1) {
						cv::circle(draw, p, 4, cv::Scalar(0, 0, 100), -1, 8, 0);//暗红色
					} else {
						cv::circle(draw, p, 4, cv::Scalar(100, 0, 0), -1, 8, 0);//深蓝色
					}
				}

				for(auto m : matches[cam]) {
					auto p1 = view_frames->keypoints_p[cam][frame][m.first];
					auto p2 = view_frames->keypoints_p[cam][frame-dframe][m.second];
					cv::arrowedLine(draw, p1, p2,
							cv::Scalar(0, 0, 0), 1, CV_AA);
				}

				for(int i=0; i<good_matches[cam].size(); i++) {
					auto m = good_matches[cam][i];
					auto p1 = view_frames->keypoints_p[cam][frame][m.first];
					auto p2 = view_frames->keypoints_p[cam][frame-dframe][m.second];
					cv::Scalar color = cv::Scalar(0, 0, 0);
					switch(residual_type[cam][i]) {
						case RESIDUAL_3D3D:
							color = cv::Scalar(0, 100, 255);//橘色
							break;
						case RESIDUAL_3D2D:
							color = cv::Scalar(255, 0, 0);//蓝色
							break;
						case RESIDUAL_2D3D:
							color = cv::Scalar(0, 230, 255);//黄色
							break;
						case RESIDUAL_2D2D:
							color = cv::Scalar(255, 200, 0);//天蓝色
							break;
					}
					cv::arrowedLine(draw, p1, p2,color, 2, CV_AA);
				}

				// Draw stereo matches
				std::vector<std::pair<int, int>> intercamera_matches;
				Solver::matchUsingId(view_frames->keypoint_ids, 0, 1, frame, frame,
						intercamera_matches);
				for(auto m : intercamera_matches) {
					auto p1 = view_frames->keypoints_p[0][frame][m.first];
					auto p2 = view_frames->keypoints_p[1][frame][m.second];
					cv::line(draw, p1, p2, cv::Scalar(255, 0, 255), 1, CV_AA);//大红色
				}
				draw.copyTo(draws[cam]);
			}
			cv::Mat D;
			vconcat(draws[0], draws[1], D);//将两张图合并成一张显示
			cv::imshow(features_windown, D);
			//cvWaitKey(1);
		}


	}

}
