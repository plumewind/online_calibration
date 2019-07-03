#include "Tools.h"

namespace online_calibration
{
    pcl::PointXYZ Tools::linterpolate(
            const pcl::PointXYZ p1,
            const pcl::PointXYZ p2,
            const float start,
            const float end,
            const float mid) {
        float a = (mid-start)/(end-start);
        float b = 1 - a;
        float x = p1.x * b + p2.x * a;
        float y = p1.y * b + p2.y * a;
        float z = p1.z * b + p2.z * a;
        return pcl::PointXYZ(x, y, z);
    }
    float Tools::linterpolate(
            const float p1,
            const float p2,
            const float start,
            const float end,
            const float mid) {
        float a = (mid-start)/(end-start);
        float b = 1 - a;
        return p1 * b + p2 * a;
    }


    void Tools::save_cloud_txt(std::string s, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
        std::ofstream fout;
        fout.open(s.c_str());
        for(int i=0; i<cloud->size(); i++) {
            fout << std::setprecision(9) << cloud->at(i).x << " "
                << cloud->at(i).y << " "
                << cloud->at(i).z << std::endl;
        }
        fout.close();
    }
    Eigen::Matrix4d Tools::pose_mat2vec(const double transform[6]) {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity(4,4);
        double R[9];
        double R_angle_axis[3];
        for(int i=0; i<3; i++) R_angle_axis[i] = transform[i];
        ceres::AngleAxisToRotationMatrix<double>(R_angle_axis, R);
        for(int j=0; j<3; j++) {
            for(int i=0; i<3; i++) {
                T(i, j) = R[j*3 + i];
            }
        }
        T(0, 3) = transform[3];
        T(1, 3) = transform[4];
        T(2, 3) = transform[5];
        return T;
    }
    void Tools::pose_vec2mat(const Eigen::Matrix4d T, double transform[6]) {
        double R[9];
        for(int j=0; j<3; j++) {
            for(int i=0; i<3; i++) {
                R[j*3 + i] = T(i, j);
            }
        }
        double R_angle_axis[3];
        ceres::RotationMatrixToAngleAxis<double>(R, R_angle_axis);
        for(int i=0; i<3; i++) transform[i] = R_angle_axis[i];
        transform[3] = T(0, 3);
        transform[4] = T(1, 3);
        transform[5] = T(2, 3);
    }
    void Tools::transform_point(pcl::PointXYZ &p, const double transform[6]) {
        double x[3] = {p.x, p.y, p.z}, y[3] = {0, 0, 0};
        ceres::AngleAxisRotatePoint(transform, x, y);
        p.x = y[0] + transform[3];
        p.y = y[1] + transform[4];
        p.z = y[2] + transform[5];
    }

    cv::Point2f Tools::geomedian(std::vector<cv::Point2f> P) {
        int m = P.size();
        cv::Point2f y(0,0);
        for(int i=0; i<m; i++) {
            y += P[i];
        }
        y /= (float)m;
        for(int iter = 0; iter < 20; iter++) {
            cv::Point2f yy(0,0);
            float d = 0;
            for(int i=0; i<m; i++) {
                float no = cv::norm(P[i] - y);
                if(no < geomedian_EPS) {
                    return y;
                }
                float nn = 1.0/no;

                yy += P[i]*nn;
                d += nn;
            }
            if(cv::norm(yy/d - y) < geomedian_EPS) {
                return yy/d;
            }
            y = yy/d;
        }
        return y;
    }


}
