#include "CameraFrame.h"

namespace online_calibration
{
	CameraFrame::CameraFrame(int num_cams_, int num_frames_)
		:num_cams(num_cams_), num_frames(num_frames_)
	{
			// tracked keypoints, camera canonical coordinates
		keypoints = std::vector<std::vector<std::vector<cv::Point2f>>>(num_cams,
				std::vector<std::vector<cv::Point2f>>(num_frames));
		// tracked keypoints, pixel coordinates
		keypoints_p = std::vector<std::vector<std::vector<cv::Point2f>>>(num_cams,
				std::vector<std::vector<cv::Point2f>>(num_frames));
		// IDs of keypoints
		keypoint_ids = std::vector<std::vector<std::vector<int>>>(num_cams,
				std::vector<std::vector<int>>(num_frames));
		// FREAK descriptors of each keypoint
		descriptors = std::vector<std::vector<cv::Mat>>(num_cams,
				std::vector<cv::Mat>(num_frames));
		// -1 if no depth, index of kp_with_depth otherwise
		has_depth = std::vector<std::vector<std::vector<int>>>(num_cams,
				std::vector<std::vector<int>>(num_frames));
		// interpolated lidar point, physical coordinates
		kp_with_depth = std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>>(num_cams,
				std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>(num_frames));
	}
	CameraFrame::~CameraFrame()
	{
	
	}
	cv::Point2f CameraFrame::pixel2canonical(
        const cv::Point2f &pp,
        const Eigen::Matrix3f &Kinv) {
		Eigen::Vector3f p;
		p << pp.x, pp.y, 1;
		p = Kinv * p;
		return cv::Point2f(p(0)/p(2), p(1)/p(2));
	}

	cv::Point2f CameraFrame::canonical2pixel(
			const cv::Point2f &pp,
			const Eigen::Matrix3f &K) {
		Eigen::Vector3f p;
		p << pp.x, pp.y, 1;
		p = K * p;
		return cv::Point2f(p(0)/p(2), p(1)/p(2));
	}


}
