#ifndef TOOLS_H
#define TOOLS_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <deque>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <list>
#include <unordered_map>
#include <random>
#include <assert.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/StdVector>
#include <Eigen/Dense>

namespace online_calibration
{
    const double INF = 1e18;
    const double PI = 3.1415926535897932384626433832795028;
    const float geomedian_EPS = 1e-6;
    const float kp_EPS = 1e-6;

	class Tools
	{
	public:
        static pcl::PointXYZ linterpolate(
            const pcl::PointXYZ p1,
            const pcl::PointXYZ p2,
            const float start,
            const float end,
            const float mid);
        static float linterpolate(
            const float p1,
            const float p2,
            const float start,
            const float end,
            const float mid);

        static inline void add_assign(pcl::PointXYZ &a, const pcl::PointXYZ &b){
            a.x += b.x;
            a.y += b.y;
            a.z += b.z;
        }

        static inline void subtract_assign(pcl::PointXYZ &a, const pcl::PointXYZ &b) {
            a.x -= b.x;
            a.y -= b.y;
            a.z -= b.z;
        }

        static inline pcl::PointXYZ add(const pcl::PointXYZ &a, const pcl::PointXYZ &b){
            return pcl::PointXYZ(a.x+b.x, a.y+b.y, a.z+b.z);
        }

        static inline void scale(pcl::PointXYZ &p, double s){
            p.x *= s;
            p.y *= s;
            p.z *= s;
        }
        
        static inline double norm2(const pcl::PointXYZ &p){
            return p.x*p.x + p.y*p.y + p.z*p.z;
        }

        static inline double norm(const pcl::PointXYZ &p){
            return std::sqrt(norm2(p));
        }

        static inline double dist2(const cv::Point2f &a, const cv::Point2f &b){
            return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
        }

        static void save_cloud_txt(std::string s, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
        static Eigen::Matrix4d pose_mat2vec(const double transform[6]);
        static void pose_vec2mat(const Eigen::Matrix4d T, double transform[6]);
        static void transform_point(pcl::PointXYZ &p, const double transform[6]);
        static cv::Point2f geomedian(std::vector<cv::Point2f> P);


		
		

	
	};

    class UF {
    public:
        UF(int n, int k) {
            _n = n;
            _k = k;
            _p = std::vector<int>(n);
            for(int i=0; i<n; i++) {
                _p[i] = i;
            }
        }
        void Union(int i, int j) {
            _p[Find(i)] = Find(j);
        }
        int Find(int i) {
            if (i > _p.size()) {
                std::cerr << "WTF how can you find something out of bounds" << std::endl;
            }
            return i == _p[i] ? i : _p[i] = Find(_p[i]);
        }
        void aggregate(std::map<int, std::set<int> > &q, int group_size) {
            for(int i=0; i<_n; i++) {
                Find(i);
            }
            for(int i=0; i<_n; i++) {
                _q[Find(i)].insert(i);
            }
            std::cerr << "Distinct classes: " << _q.size() << std::endl;
            int groups = 0;
            for(std::map<int, std::set<int> >::iterator it = _q.begin();
                    it != _q.end();
                    it++) {
                //std::cerr << "Class of " << it->second.size() << std::endl;
                if(it->second.size() == group_size/2) {
                    for(std::set<int>::iterator itt = it->second.begin();
                            itt != it->second.end();
                            itt++) {
                        q[groups].insert(*itt);
                    }
                    groups++;
                }
            }
            std::cerr << "groups: " << groups << std::endl;
        }
    private:
        int _n, _k;
        std::vector<int> _p;
        std::map<int, std::set<int> > _q;
};



}

#endif
