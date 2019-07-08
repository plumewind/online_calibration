#ifndef VIEWER_H
#define VIEWER_H

#include "Tools.h"
#include "CameraFrame.h"
#include "Tracking.h"
#include "Solver.h"

namespace online_calibration
{
	class CameraFrame;
	class Tracking;

	class Viewer
	{
	public:
		Viewer(int num_cams_, CameraFrame* view_frames_mp, Tracking* view_track_mp, Solver* view_solver_mp);
		~Viewer();
		
		void view_depth_assoc(cv::Mat &imgs, int cam, int  frame,
				std::vector<std::vector<cv::Point2f>>& projection,
                std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& scans_valid);
		void view_features(int frame, int dframe, std::vector<cv::Mat>& imgs,
						std::vector<std::vector<std::pair<int, int>>>& matches,
						std::vector<std::vector<std::pair<int, int>>>& good_matches,
						std::vector<std::vector<ResidualType>>& residual_type);

	private:
		int num_cams;
		char* features_windown;
		char* depthassoc_windown;

		CameraFrame* view_frames;
		Tracking* view_track;
		Solver* view_solver;
	};
}

#endif
