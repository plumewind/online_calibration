#ifndef SYSTEM_H
#define SYSTEM_H

#include "Tracking.h"
#include "CameraFrame.h"
#include "Solver.h"
#include "Viewer.h"
#include "Tools.h"

//#define MatZero34 Eigen::MatrixXf::Zero(3, 4)
Eigen::Matrix4f MatZero44 = Eigen::MatrixXf::Zero(4, 4);

namespace online_calibration
{
	class Tracking;
	class CameraFrame;
	class Solver;
	class Viewer;

	class System
	{
	public:
		System(int num_cams_, int num_frames_, std::string file_name_);
		~System();
		
		void system_init(std::vector<Eigen::Matrix<float, 3, 4>,
    						Eigen::aligned_allocator<Eigen::Matrix<float, 3, 4>>>& cam_mat,
						Eigen::Matrix4f& velo2cam);
		void run(const cv::Mat &left_img, 
				const cv::Mat &right_img,
				const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans, 
				int frame
				);

		void run(const cv::Mat &img, 
				const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans, 
				int frame);
		

	private:
		std::string file_name;

		std::vector<cv::Mat> img_prevs;
		int num_cams, num_frames; // number of cameras we use
		int min_matches; // minimum number of feature matches to proceed
		// attempt matching frames between dframes
		std::vector<std::vector<int>> dframes;
		int ndiagonal,
			detect_every; // detect new features every this number of frames
		
		// counters for keypoint id, one for each camera
    	int id_counter;

		double	loop_close_thresh, // meters
				agreement_t_thresh , // meters
				agreement_r_thresh ; // radians

		Tracking* sys_Tracking;
		CameraFrame* sys_Frames;
		Solver* sys_Solver;
		Viewer* sys_Viewer;
	};
}

#endif
