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

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <ctime>
#include <fstream>

std::vector<double> gftt_times;
std::vector<double> orb_times;
std::vector<double> fast_times;

//话题回调函数
void rgb_image_callback(const sensor_msgs::Image::ConstPtr& msg)
{
	//ROS_INFO("node_b is receiving [%s]", msg->data.c_str());
    
	cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat imageSource,  gray_image;
  imageSource=cv_ptr->image;
  cv::cvtColor(imageSource, gray_image, CV_BGR2GRAY);

  //检测gftt特帧点
  double max_corners=100;
  std::vector<cv::Point2f> corners;
  clock_t time_stt = clock(); // 计时
	cv::goodFeaturesToTrack(gray_image, corners, max_corners, 0.01, 10);
  double time_cost = 1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC;
  gftt_times.push_back(time_cost);
  std::cout <<"<1>time use in goodFeaturesToTrack is " << (time_cost) << "ms"<< std::endl;

  //检测orb特征点
  std::vector<cv::KeyPoint> keypoints1;
  cv::Mat descriptors1;
  cv::Ptr<cv::ORB> orb_detector = cv::ORB::create(max_corners);
	//提取 Oriented FAST 特征点
  time_stt = clock(); // 计时
	orb_detector->detect(gray_image, keypoints1);
  time_cost = 1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC;
  orb_times.push_back(time_cost);
  std::cout <<"<2>time use in orb_detector is " << (time_cost) << "ms"<< std::endl;
	//根据角点位置计算 BRIEF 描述子
	orb_detector->compute(gray_image, keypoints1, descriptors1);
  std::vector<cv::Point2f> points1;
  cv::KeyPoint::convert(keypoints1, points1);

  //检测fast特征点
  std::vector<cv::KeyPoint> descriptors2;
  cv::Ptr<cv::FastFeatureDetector> fast_detector = cv::FastFeatureDetector::create(max_corners);
  time_stt = clock(); // 计时
  fast_detector->detect(gray_image, descriptors2);
  time_cost = 1000* (clock() - time_stt)/(double)CLOCKS_PER_SEC;
  fast_times.push_back(time_cost);
  std::cout <<"<3>time use in fast_detector is " << (time_cost) << "ms"<< std::endl;
  std::vector<cv::Point2f> points2;
  cv::KeyPoint::convert(descriptors2, points2);

  //特征点显示
	for(int i=0; i<corners.size(); i++)
	{
		cv::circle(imageSource,corners[i],2,cv::Scalar(0,0,255),2);//红色
    cv::circle(imageSource,points1[i],2,cv::Scalar(0,255,0),2);//绿色
    cv::circle(imageSource,points2[i],2,cv::Scalar(155,0,0),2);//蓝色
	}
	cv::imshow("gftt detect",imageSource);
}


int main(int argc, char **argv)
{
	// Set up ROS node
  ros::init(argc,  argv, "gftt_test");
  
  ros::NodeHandle handle;
  ros::Subscriber rgb_image_sub = handle.subscribe("/kitti/camera_color_left/image_raw", 100, rgb_image_callback);
  
  std::ofstream result_file;                                                   //定义输出文件
  result_file.open("/home/user/catkin_vloam/src/vloam/test/times_cost_detect.txt");     //作为输出文件打开

  cv::namedWindow("gftt detect");
  cv::startWindowThread();
  
  ROS_INFO("gftt_test is beginning......");
  ros::spin();
  
  result_file<<"The cost of times in features detected."<<std::endl;
  for(int i = 0 ; i < gftt_times.size() ; i++)
    result_file<<gftt_times[i]<<" ";
  result_file<<std::endl;

  for(int i = 0 ; i < orb_times.size() ; i++)
    result_file<<orb_times[i]<<" ";
  result_file<<std::endl;

  for(int i = 0 ; i < fast_times.size() ; i++)
    result_file<<fast_times[i]<<" ";
  result_file<<std::endl;

  result_file.close();

  ROS_INFO("result_file is over......");
	return 0;
}
