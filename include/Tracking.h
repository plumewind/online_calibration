#ifndef TRACKING_H
#define TRACKING_H

#include "Tools.h"
#include "CameraFrame.h"
//#include "./exmaples/kitti/kitti_odometry.h"

namespace online_calibration
{
	class CameraFrame;

	class Tracking
	{
	public:
		Tracking(int num_cams_, CameraFrame* track_frames_mp);
		~Tracking();
		
		void setCameraKinv(
				Eigen::Matrix4f& velo2cam,
				std::vector<Eigen::Matrix<float, 3, 4>,
    			Eigen::aligned_allocator<Eigen::Matrix<float, 3, 4>>>& cam_mat);

		void trackFeatures(
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
			);
		void detectFeatures(
			//std::vector<std::vector<cv::Point2f>> &keypoints,
			//std::vector<std::vector<cv::Point2f>> &keypoints_p,
			//std::vector<std::vector<int>> &keypoint_ids,
			//std::vector<cv::Mat> &descriptors,
			//const cv::Ptr<cv::FeatureDetector> detector,
			//const cv::Ptr<cv::DescriptorExtractor> extractor,
			const cv::Mat &img,
			int &id_counter,
			const int cam,
			const int frame
			);

		void consolidateFeatures(
			// std::vector<cv::Point2f> &keypoints,
			// std::vector<cv::Point2f> &keypoints_p,
			// std::vector<int> &keypoint_ids,
			// cv::Mat &descriptors,
			const int cam,
			const int frame
		);
		void removeTerribleFeatures(
			//std::vector<cv::Point2f> &keypoints,
			//std::vector<cv::Point2f> &keypoints_p,
			//std::vector<int> &keypoint_ids,
			//cv::Mat &descriptors,
			//const cv::Ptr<cv::DescriptorExtractor> extractor,
			const cv::Mat &img,
			const int cam,
			const int frame
        );
		void projectLidarToCamera(
			const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans,
			std::vector<std::vector<cv::Point2f>> &projection,
			std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans_valid,
			const int cam
        );
		void featureDepthAssociation(
			const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans,//合格激光点云
			const std::vector<std::vector<cv::Point2f>> &projection,      //合格激光点的投影点
			//const std::vector<cv::Point2f> &keypoints,                    //图像特征点
			//pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_with_depth,
			//std::vector<int> &has_depth
			const int cam,
			const int frame
        );
		void removeSlightlyLessTerribleFeatures(
        //std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints,
        //std::vector<std::vector<std::vector<cv::Point2f>>> &keypoints_p,
        //std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> &kp_with_depth,
        //std::vector<std::vector<std::vector<int>>> &keypoint_ids,
        //std::vector<std::vector<cv::Mat>> &descriptors,
        //std::vector<std::vector<std::vector<int>>> &has_depth,
        const int frame,
        const std::vector<std::vector<std::pair<int, int>>> &good_matches);

		// std::vector<Eigen::Matrix<float, 3, 4>,
		// 	Eigen::aligned_allocator<Eigen::Matrix<float, 3, 4>>> cam_mat;//相机的信息矩阵，包含R和t
		std::vector<Eigen::Matrix3f,
			Eigen::aligned_allocator<Eigen::Matrix3f>> cam_intrinsic;//相机内参矩阵
		std::vector<Eigen::Matrix3f,
			Eigen::aligned_allocator<Eigen::Matrix3f>> cam_intrinsic_inv;
		std::vector<Eigen::Vector3f,
			Eigen::aligned_allocator<Eigen::Vector3f>> cam_trans;//相机相对于相机0的偏移矩阵
		std::vector<Eigen::Matrix4f,
			Eigen::aligned_allocator<Eigen::Vector3f>> cam_pose;//相机的位姿

	private:
		long double	flow_outlier; // pixels^2, squared distance of optical flow
    	double	quality_level; // good features to track quality
    	double	min_distance; // pixel distance between nearest features
		double	lkt_window;
		double	lkt_pyramid;
		double	match_thresh;// bits, hamming distance for FREAK features
		double	depth_assoc_thresh; // canonical camera units
		double	corner_count; // number of features

		int img_width, // kitti 数据集中图像的宽度和高度
    		img_height;

		int num_cams;

		Eigen::Matrix4f velo_to_cam, cam_to_velo;       //激光，相机之间的坐标转换关系
		std::vector<double> min_x, max_x, min_y, max_y;//激光点云在图像上的有效投影区域

		// FREAK feature descriptor
    	cv::Ptr<cv::xfeatures2d::FREAK> track_freak;
		// good features to detect
    	cv::Ptr<cv::GFTTDetector> track_gftt;

		CameraFrame* track_frames;
	};
}

#endif
