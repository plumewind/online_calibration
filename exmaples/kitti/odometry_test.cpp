#include "System.h"
#include "kitti_odometry.h"

#define USE_CUDA

int main(int argc, char** argv)
{
	cv::setUseOptimized(true);

#ifdef USE_CUDA
    cv::cuda::setDevice(0);
#endif

	if(argc < 2) {
        std::cout << "Usage: velo kitti_dataset_number. e.g: ./build/velo 00" << std::endl;
        return 1;
    }
    std::string dataset = argv[1];
    
	loadImage(dataset, 0, 0);// to set width and height
    loadCalibration(dataset);
    loadTimes(dataset);
	
    const int num_cams = 2; // number of cameras we use
	// images of current and frame, used for optical flow tracking
    std::vector<cv::Mat> imgs(num_cams);
    std::vector<cv::Mat> img_prevs(num_cams);
    ScansLRU lru;

	int num_frames = times.size();
    online_calibration::System cam2lidar_calib(num_cams, num_frames, std::string(argv[1]));
    cam2lidar_calib.system_init(cam_mat, velo_to_cam);

    std::cout << "test1"<<std::endl;
	for(int frame = 0; frame < num_frames; frame++) //枚举所有帧
	{
		ScanData *sd = lru.get(dataset, frame);//提取激光点云数据，并做预处理
        const auto &scans = sd->scans;
        for(int cam = 0; cam<num_cams; cam++) {//提取当前摄像头的图像
            imgs[cam] = loadImage(dataset, cam, frame);
        }
        std::cout << "test2"<<std::endl;
		//传入图像数据和激光数据进行处理
		cam2lidar_calib.run(imgs[0], imgs[1],  scans, frame);
	}
	return 0;
}
