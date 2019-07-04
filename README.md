# online_calibration
This is an online calibration system between multiple sensors (camera, lidar, IMU). It is being created. . . . . .

# from
Vision-Enhanced Lidar Odometry and Mapping (VELO) is a new algorithm for simultaneous localization and mapping using a set of cameras and a lidar. By tightly coupling sparse visual odometry and lidar scan matching, VELO is able to achieve reduced drift error compared to existing state-of-the-art algorithms for visual-lidar pose estimation. Moreover, the algorithm is capable of functioning when either the lidar or the camera is blinded. Experimental results are demonstrated using the KITTI data set as well as our own data set obtained using an off-road vehicle.

This code is currently work in progress and is not ready to be run. Use at your own risk.

## Requirements

* iSAM 1.7
* OpenCV (latest git version, compile from source)
* OpenCV contrib (xfeatures2d)
* PCL 1.7.2
* Ceres Solver
* CUDA 7.5

## References
Lu D L. Vision-enhanced lidar odometry and mapping[D]. Carnegie Mellon University Pittsburgh, PA, 2016.
