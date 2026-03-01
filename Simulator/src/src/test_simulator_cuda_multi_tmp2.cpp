#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <pcl_ros/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "sensor_simulator.cuh"
#include <chrono>
#include "maps.hpp"

using namespace raycast;

GridMap* grid_map = nullptr;  // 声明全局变量
sensor_msgs::PointCloud2 output;
ros::Publisher pcl_pub;

class SensorSimulator {
    public:
        SensorSimulator(ros::NodeHandle &nh, const std::string &namespace_) 
            : nh_(nh){
            YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);
    
            std::string odom_topic = namespace_ + "/odom";
            std::string depth_topic = namespace_ + "/depth";
            std::string lidar_topic = namespace_ + "/lidar";
    
            // ROS
            image_pub_ = nh_.advertise<sensor_msgs::Image>(depth_topic, 1);
            point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_topic, 1);
            odom_sub_ = nh_.subscribe(odom_topic, 1, &SensorSimulator::odomCallback, this);
    
            render_depth = config["render_depth"].as<bool>();
            render_lidar = config["render_lidar"].as<bool>();
            float depth_fps = config["depth_fps"].as<float>();
            float lidar_fps = config["lidar_fps"].as<float>();
    
            // timer_depth_ = nh_.createTimer(ros::Duration(1 / depth_fps), &SensorSimulator::timerDepthCallback, this);
            timer_lidar_ = nh_.createTimer(ros::Duration(1 / lidar_fps), &SensorSimulator::timerLidarCallback, this);
        }
    
        void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);
        void timerDepthCallback(const ros::TimerEvent &);
        void timerLidarCallback(const ros::TimerEvent &);
    
    private:
        bool render_depth{false};
        bool render_lidar{false};
        bool odom_init{false};
        Eigen::Quaternionf quat;
        Eigen::Quaternionf quat_bc, quat_wc;
        Eigen::Vector3f pos;

        extern GridMap* grid_map;  // 声明全局变量
    
        CameraParams* camera;
        LidarParams* lidar;
    
        ros::NodeHandle nh_;
        ros::Publisher image_pub_, point_cloud_pub_;
        ros::Subscriber odom_sub_;
        ros::Timer timer_depth_, timer_lidar_;
};


void SensorSimulator::timerDepthCallback(const ros::TimerEvent&) {
    if (!odom_init || !render_depth) {
        std::cerr << "Warning: Depth rendering skipped due to initialization issues." << std::endl;
        return;
    }

    if (!grid_map || !camera) {
        std::cerr << "Error: GridMap or CameraParams is not initialized!" << std::endl;
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat_wc.w(), quat_wc.x(), quat_wc.y(), quat_wc.z(), pos.x(), pos.y(), pos.z());
    cv::Mat depth_image;
    std::cout << "Rendering depth image..." << std::endl;
    renderDepthImage(grid_map, camera, T_wc, depth_image);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "生成图像耗时: " << elapsed.count() << " 秒" << std::endl;

    sensor_msgs::Image ros_image;
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = ros::Time::now();
    cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_image.image = depth_image;
    cv_image.toImageMsg(ros_image);
    image_pub_.publish(ros_image);
}


void SensorSimulator::timerLidarCallback(const ros::TimerEvent&) {
    if (!odom_init || !render_lidar)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat.w(), quat.x(), quat.y(), quat.z(), pos.x(), pos.y(), pos.z());
    pcl::PointCloud<pcl::PointXYZ> lidar_points;
    std::cout << "Rendering LiDAR point cloud..." << std::endl;
    renderLidarPointcloud(grid_map, lidar, T_wc, lidar_points);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "生成雷达耗时: " << elapsed.count() << " 秒" << std::endl;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(lidar_points, output);
    output.header.stamp = ros::Time::now();
    output.header.frame_id = "odom";
    point_cloud_pub_.publish(output);
}

void SensorSimulator::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    quat.x() = msg->pose.pose.orientation.x;
    quat.y() = msg->pose.pose.orientation.y;
    quat.z() = msg->pose.pose.orientation.z;
    quat.w() = msg->pose.pose.orientation.w;
    quat_wc = quat * quat_bc;

    pos.x() = msg->pose.pose.position.x;
    pos.y() = msg->pose.pose.position.y;
    pos.z() = msg->pose.pose.position.z;

    odom_init = true;
    std::cout << "Odom init: " << pos.x() << ", " << pos.y() << ", " << pos.z() << std::endl;
}

void timerMapCallback(const ros::TimerEvent&) {
    if (!pcl_pub) {
        std::cerr << "Error: pcl_pub is not initialized!" << std::endl;
        return;
    }
    if (pcl_pub.getNumSubscribers() > 0)
        pcl_pub.publish(output);    
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_simulator_node");
    ros::NodeHandle nh;

    // 加载配置文件
    YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);
    float resolution = config["resolution"].as<float>();
    int occupy_threshold = config["occupy_threshold"].as<int>();
    bool use_random_map = config["random_map"].as<bool>();
    std::string ply_file = config["ply_file"].as<std::string>();
    int seed = config["seed"].as<int>();
    int sizeX = config["x_length"].as<int>();
    int sizeY = config["y_length"].as<int>();
    int sizeZ = config["z_length"].as<int>();
    int type = config["maze_type"].as<int>();
    double scale = 1 / resolution;
    sizeX = sizeX * scale;
    sizeY = sizeY * scale;
    sizeZ = sizeZ * scale;

    // 生成地图
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (use_random_map) {
        printf("1.Generate Random Map... \n");
        mocka::Maps::BasicInfo info;
        info.sizeX      = sizeX;
        info.sizeY      = sizeY;
        info.sizeZ      = sizeZ;
        info.seed       = seed;
        info.scale      = scale;
        info.cloud      = cloud;

        mocka::Maps map;
        map.setParam(config);
        map.setInfo(info);
        map.generate(type);
    } else {
        printf("1.Reading Point Cloud %s... \n", ply_file.c_str());
        if (pcl::io::loadPLYFile(ply_file, *cloud) == -1) {
            PCL_ERROR("Couldn't read PLY file \n");
        }
    }

    pcl::toROSMsg(*cloud, output);
    output.header.frame_id = "world";

    std::cout<<"Pointloud size:"<<cloud->points.size()<<std::endl;
    printf("2.Mapping... \n");

    grid_map = new GridMap(cloud, resolution, occupy_threshold);

    ros::Timer timer_map_;
    timer_map_   = nh.createTimer(ros::Duration(1), &timerMapCallback);

    // 创建多个无人机的传感器模拟器实例
    SensorSimulator sensor_simulator_1(nh, "drone1");
    // SensorSimulator sensor_simulator_2(nh, shared_map, "drone2");

    printf("3.Simulation Ready! \n");

    ros::spin();
    return 0;
}