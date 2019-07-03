#ifndef CAMERAFRAME_H
#define CAMERAFRAME_H

#include "Tools.h"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>

namespace online_calibration
{
	class CameraFrame
	{
	public:
		CameraFrame(int num_cams_, int num_frames_);
		~CameraFrame();

		static cv::Point2f pixel2canonical(
			const cv::Point2f &pp,
			const Eigen::Matrix3f &Kinv);
		static cv::Point2f canonical2pixel(
			const cv::Point2f &pp,
			const Eigen::Matrix3f &K);


		// tracked keypoints, camera canonical coordinates
		std::vector<std::vector<std::vector<cv::Point2f>>> keypoints;
		// tracked keypoints, pixel coordinates
		std::vector<std::vector<std::vector<cv::Point2f>>> keypoints_p;
		// IDs of keypoints
		std::vector<std::vector<std::vector<int>>> keypoint_ids;
		// FREAK descriptors of each keypoint
		std::vector<std::vector<cv::Mat>> descriptors;
		// -1 if no depth, index of kp_with_depth otherwise
		std::vector<std::vector<std::vector<int> > > has_depth;
		// interpolated lidar point, physical coordinates
		std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> kp_with_depth;

		// vector which maps from keypoint id to observed position
		// [keypoint_id][cam][frame] = observation
		std::vector<std::vector<std::map<int, cv::Point2f>>> keypoint_obs2;
		// [keypoint_id][cam][frame] = observation3
		std::vector<std::vector<std::map<int, pcl::PointXYZ>>> keypoint_obs3;
		// number of observations for each keypoint_id
		std::vector<int> keypoint_obs_count;
		
	private:
		int num_cams;
		int num_frames;
	};
}

#endif
