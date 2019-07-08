//#include "vloam/system.h"
//#include "vloam/tracking.h"

//ROS部分
#include <ros/ros.h>
#include <ros/package.h>

#include <sensor_msgs/Image.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <ctime>
#include <fstream>
#include <iostream>

std::vector<double> gftt_times;
std::vector<double> orb_times;
std::vector<double> fast_times;

bool initial_flag=false;
int img_width, img_height;
cv::Mat prevFrame;
const double corner_count=300;
const double quality_level = 0.001;	// good features to track quality
const double min_distance = 12;		// pixel distance between nearest features
const double lkt_window = 21;
const double lkt_pyramid = 4;
const double flow_outlier = 20000;	// pixels^2, squared distance of optical flow

//话题回调函数
void rgb_image_callback(const sensor_msgs::Image::ConstPtr& msg)
{
	//ROS_INFO("node_b is receiving [%s]", msg->data.c_str());
    
	cv_bridge::CvImagePtr cv_ptr;
	try{
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	}catch (cv_bridge::Exception& e){
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	cv::Mat prevFrame_gray,  nextFrame_gray;
	cv::Mat nextFrame = cv_ptr->image;
	if(initial_flag == false)
	{
		prevFrame = nextFrame;
		initial_flag = true;
		img_width = prevFrame.cols;
    	img_height = prevFrame.rows;
		return ;
	}
	cv::cvtColor(prevFrame, prevFrame_gray, CV_BGR2GRAY);
	cv::cvtColor(nextFrame, nextFrame_gray, CV_BGR2GRAY);

	//检测GFTT特帧点+FREAK描述子
    cv::Ptr<cv::xfeatures2d::FREAK> freak_extractor = cv::xfeatures2d::FREAK::create(// FREAK feature descriptor
            false, // orientation normalization
            false // scale normalization
            );
    cv::Ptr<cv::GFTTDetector> gftt_detector = cv::GFTTDetector::create(// good features to detect
            corner_count,
            quality_level,
            min_distance);

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	std::vector<unsigned char> status;
    std::vector<float> err;
	std::vector<cv::Point2f> points1, points2, points_good;

	clock_t time_stt = clock(); // 计时开始
    gftt_detector->detect(prevFrame_gray, keypoints1);
	gftt_detector->detect(nextFrame_gray, keypoints2);
	
    freak_extractor->compute(prevFrame_gray, keypoints1, descriptors1);
	freak_extractor->compute(nextFrame_gray, keypoints2, descriptors2);
	
	//用LK光流做特征点匹配
	cv::KeyPoint::convert(keypoints1, points1);
  	cv::KeyPoint::convert(keypoints2, points2);
    cv::calcOpticalFlowPyrLK(prevFrame_gray, nextFrame_gray, points1, points2,
            status, err,
            cv::Size(lkt_window, lkt_window),
            lkt_pyramid,
            cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01), 0);
	
	for(int i=0; i<status.size(); i++) {
		if(!status[i]) 
			continue;
		if((std::pow(points1[i].x - points2[i].x, 2) + std::pow(points1[i].y-points2[i].y, 2)) > flow_outlier) 
			continue;
	
		// somehow points can be tracked to negative x and y
		if(points2[i].x < 0 || points2[i].y < 0 ||
			points2[i].x >= img_width ||
			points2[i].y >= img_height) 
			continue;

		points_good.push_back(points2[i]);
	}
	double time_cost = 1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC;//计算消耗时间
	gftt_times.push_back(time_cost);
	for(int i=0; i<points_good.size(); i++)//特征点显示
		cv::circle(nextFrame,points_good[i],2,cv::Scalar(0,0,255),2);//红色
	cv::imshow("gftt detect", nextFrame);
	std::cout <<"<1>time use in goodFeaturesToTrack is " << (time_cost) << "ms"<< std::endl;

	//检测orb特征点
	cv::Ptr<cv::ORB> orb_detector = cv::ORB::create(corner_count);
	cv::Ptr<cv::DescriptorExtractor> orb_extractor = cv::ORB::create(corner_count);
	cv::Ptr<cv::DescriptorMatcher> orb_matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );

	time_stt = clock(); // 计时
	orb_detector->detect ( prevFrame_gray, keypoints1 );//检测 Oriented FAST 角点
    orb_detector->detect ( nextFrame_gray, keypoints2 );
	
	orb_extractor->compute ( prevFrame_gray, keypoints1, descriptors1 );//计算 BRIEF 描述子
    orb_extractor->compute ( nextFrame_gray, keypoints2, descriptors2 );
	
	
	std::vector<cv::DMatch> orb_matches;  
    orb_matcher->match ( descriptors1, descriptors2, orb_matches );//BFMatcher matcher ( NORM_HAMMING );
	
	double min_dist=10000, max_dist=0;
	for ( int i = 0; i < descriptors1.rows; i++ )
    {
        double dist = orb_matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

	//min_dist = std::min_element( orb_matches.begin(), orb_matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) {return m1.distance<m2.distance;} )->distance;
    //max_dist = std::max_element( orb_matches.begin(), orb_matches.end(), [](const cv::DMatch& m1, const cv::DMatch& m2) {return m1.distance<m2.distance;} )->distance;

	std::vector< cv::DMatch > good_matches;
    for ( int i = 0; i < descriptors1.rows; i++ )
    {
        if ( orb_matches[i].distance <= std::max ( 2*min_dist, 30.0 ) )
            good_matches.push_back ( orb_matches[i] );
    }
	time_cost = 1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC;
	orb_times.push_back(time_cost);

	// printf ( "-- Max dist : %f \n", max_dist );
    // printf ( "-- Min dist : %f \n", min_dist );
	std::cout <<"<2>time use in orb_detector is " << (time_cost) << "ms"<< std::endl;
	cv::Mat frame_goodmatch;
	cv::drawMatches ( prevFrame, keypoints1, nextFrame, keypoints2, good_matches, frame_goodmatch);
    cv::imshow ( "orb detect", frame_goodmatch );
	prevFrame = nextFrame;
}


int main(int argc, char **argv)
{
	// Set up ROS node
  ros::init(argc,  argv, "descriptor_test");
  
  ros::NodeHandle handle;
  ros::Subscriber rgb_image_sub = handle.subscribe("/kitti/camera_color_left/image_raw", 100, rgb_image_callback);
  
  std::ofstream result_file;                                                   //定义输出文件
  result_file.open("/home/user/catkin_vloam/src/vloam/test/times_cost_match.txt");     //作为输出文件打开

  cv::namedWindow("gftt detect");
  cv::namedWindow("orb detect");
  cv::startWindowThread();
  
  ROS_INFO("gftt_test is beginning......");
  ros::spin();
  
  result_file<<"The cost of times in features matched."<<std::endl;
  for(int i = 0 ; i < gftt_times.size() ; i++)
    result_file<<gftt_times[i]<<" ";
  result_file<<std::endl;

  for(int i = 0 ; i < orb_times.size() ; i++)
    result_file<<orb_times[i]<<" ";
  result_file<<std::endl;

//   for(int i = 0 ; i < fast_times.size() ; i++)
//     result_file<<fast_times[i]<<" ";
//   result_file<<std::endl;

  result_file.close();

  ROS_INFO("result_file is over......");
	return 0;
}
