#pragma once

const int   num_cams_actual = 4; // number of cameras actually available in dataset

std::vector<Eigen::Matrix<float, 3, 4>,
    Eigen::aligned_allocator<Eigen::Matrix<float, 3, 4>>> cam_mat;//相机的信息矩阵，包含R和t

Eigen::Matrix4f velo_to_cam, cam_to_velo;       //激光，相机之间的坐标转换关系

// std::ofstream output;

std::vector<double> times;

const std::string kittipath = "/home/user/data/kitti/odometry/dataset/sequences/";
//const std::string kittipath = "/home/dllu/kitti/dataset/sequences/";

void loadCalibration(
        const std::string & dataset
        ) {
    std::string calib_path = kittipath + dataset + "/calib.txt";
    std::ifstream calib_stream(calib_path);
    std::string P;
    velo_to_cam = Eigen::Matrix4f::Identity();
    for(int cam=0; cam<num_cams_actual; cam++) {
        calib_stream >> P;
        cam_mat.push_back(Eigen::Matrix<float, 3, 4>());
        for(int i=0; i<3; i++) {
            for(int j=0; j<4; j++) {
                calib_stream >> cam_mat[cam](i,j);
            }
        }
    }
    calib_stream >> P;
    for(int i=0; i<3; i++) {
        for(int j=0; j<4; j++) {
            calib_stream >> velo_to_cam(i,j);
        }
    }

    // cam_to_velo = velo_to_cam.inverse();
}

void loadTimes(
        const std::string & dataset
        ) {
    std::string time_path = kittipath + dataset + "/times.txt";
    std::ifstream time_stream(time_path);
    double t;
    while(time_stream >> t) {
        times.push_back(t);
    }
}

void loadPoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
        std::string dataset,
        int n
        ) {
    std::stringstream ss;
    ss << kittipath << dataset << "/velodyne/"
        << std::setfill('0') << std::setw(6) << n << ".bin";
    // allocate 40 MB buffer (only ~1300*4*4 KB are needed)

    int32_t num = 10000000;
    float *data = (float*)malloc(num*sizeof(float));

    // pointers
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;

    // load point cloud
    FILE *stream;
    stream = fopen (ss.str().c_str(),"rb");
    num = fread(data,sizeof(float),num,stream)/4;


    for (int32_t i=0; i<num; i++) {
        point_cloud->points.push_back(pcl::PointXYZ(*px,*py,*pz));
        px+=4; py+=4; pz+=4; pr+=4;
    }

    fclose(stream);
    free(data);
}

void segmentPoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &scans
        ) {
    float prev_y = 0;
    int scan_id = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(
            new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*point_cloud, *cloud_tmp, velo_to_cam);//将点云从激光坐标系转换成相机坐标系
    std::vector<std::vector<int>> scan_ids;
    for(int i=0, _i = point_cloud->size(); i<_i; i++) {//枚举每一线激光点云
        pcl::PointXYZ p = point_cloud->at(i);
        if(i > 0 && p.x > 0 && (p.y > 0) != (prev_y > 0)) {//筛选合格点
            scan_id++;
        }
        if(scan_id >= scans.size()) {               //准备容器
            scan_ids.push_back(std::vector<int>());
            scans.push_back(pcl::PointCloud<pcl::PointXYZ>::Ptr(
                        new pcl::PointCloud<pcl::PointXYZ>));
        }
        scan_ids[scan_id].push_back(i);
        prev_y = p.y;
    }
    // for some reason, kitti scans are sorted in a strange way
    for(int s=0; s<scan_ids.size(); s++) {
        for(int i = 0, _i = scan_ids[s].size(); i<_i; i++) {
            pcl::PointXYZ q = cloud_tmp->at(scan_ids[s][_i - 1 - (i + _i/2) % _i]);
            scans[s]->push_back(q);
        }
    }
    cloud_tmp.reset();
}

cv::Mat loadImage(
        const std::string & dataset,
        const int cam,
        const int n
        ) {
    std::stringstream ss;
    ss << kittipath << dataset << "/image_" << cam << "/"
        << std::setfill('0') << std::setw(6) << n << ".png";
    cv::Mat I = cv::imread(ss.str(), 0);
    // img_width = I.cols;
    // img_height = I.rows;
    return I;
}


// Least-recently used cache for storing lidar scans
// because we can't keep all 5000 in memory.
// Each scan has 130,000 points, each taking up 16 bytes
// not counting the duplication and overhead in the kd tree
struct ScanData {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scans;
    std::vector<pcl::KdTreeFLANN<pcl::PointXYZ>> trees;
    int _frame;
    ScanData() {}
    ScanData(const std::string dataset, const int frame) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
                new pcl::PointCloud<pcl::PointXYZ>);
        loadPoints(cloud, dataset, frame);  //提取点云数据
        segmentPoints(cloud, scans);        //将64线点云数据分割成64组scan数据，且由激光坐标系转换到相机坐标系
        trees.resize(scans.size());
        for(int i=0; i<scans.size(); i++) {
            trees[i].setInputCloud(scans[i]);
        }
        _frame = frame;
        /*
        std::cerr << "created scandata: " << dataset 
            << ", " << frame 
            << ": " << scans.size()
            << std::endl;
            */
    }
};

class ScansLRU {
    private:
    const int size = 50;
    std::list<ScanData*> times;
    std::unordered_map<int, decltype(times)::iterator> exists;
    public:
    ScanData* get(const std::string dataset,
            const int frame
            ) {
        // retrieves from scan if possible,
        // loads data from disk otherwise
        if(exists.count(frame)) {
            auto it = exists[frame];
            ScanData *sd = *it;
            times.erase(it);
            times.push_front(sd);
            return sd;
        } else {
            ScanData *sd = new ScanData(dataset, frame);
            times.push_front(sd);
            exists[frame] = times.begin();
            if(times.size() > size) {
                auto sd = times.back();
                exists.erase(sd->_frame);
                delete sd;
                times.pop_back();
            }
            return sd;
        }
    }
};
