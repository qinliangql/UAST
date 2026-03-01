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

class SensorSimulator {
public:
    SensorSimulator(ros::NodeHandle &nh) : nh_(nh) {
        YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);
        // 读取camera参数
        camera = new CameraParams();
        camera->fx = config["camera"]["fx"].as<float>();
        camera->fy = config["camera"]["fy"].as<float>();
        camera->cx = config["camera"]["cx"].as<float>();
        camera->cy = config["camera"]["cy"].as<float>();
        camera->image_width = config["camera"]["image_width"].as<int>();
        camera->image_height = config["camera"]["image_height"].as<int>();
        camera->max_depth_dist = config["camera"]["max_depth_dist"].as<float>();
        camera->normalize_depth = config["camera"]["normalize_depth"].as<bool>();
        float pitch = config["camera"]["pitch"].as<float>() * M_PI / 180.0;
        quat_bc = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());
        quat_bc2 = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());

        // 读取lidar参数
        lidar = new LidarParams();
        lidar->vertical_lines = config["lidar"]["vertical_lines"].as<int>();
        lidar->vertical_angle_start = config["lidar"]["vertical_angle_start"].as<float>();
        lidar->vertical_angle_end = config["lidar"]["vertical_angle_end"].as<float>();
        lidar->horizontal_num = config["lidar"]["horizontal_num"].as<int>();
        lidar->horizontal_resolution = config["lidar"]["horizontal_resolution"].as<float>();
        lidar->max_lidar_dist = config["lidar"]["max_lidar_dist"].as<float>();

        render_lidar = config["render_lidar"].as<bool>();
        render_depth = config["render_depth"].as<bool>();
        float depth_fps = config["depth_fps"].as<float>();
        float lidar_fps = config["lidar_fps"].as<float>();
        
        std::string ply_file = config["ply_file"].as<std::string>();
        std::string odom_topic = "/drone1"+config["odom_topic"].as<std::string>();
        std::string odom_topic2 = "/drone2"+config["odom_topic"].as<std::string>();
        std::string depth_topic = "/drone1"+config["depth_topic"].as<std::string>();
        std::string rgb_topic = "/drone1"+config["rgb_topic"].as<std::string>();
        std::string depth_topic2 = "/drone2"+config["depth_topic"].as<std::string>();
        std::string rgb_topic2 = "/drone2"+config["rgb_topic"].as<std::string>();
        std::string lidar_topic = "/drone1"+config["lidar_topic"].as<std::string>();    // 只需要target 的lidar就行,TODO,lidar最后的结果为重影

        // 读取地图参数
        bool use_random_map = config["random_map"].as<bool>();
        float resolution = config["resolution"].as<float>();
        int occupy_threshold = config["occupy_threshold"].as<int>();
        pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("mock_map", 1);
        int seed = config["seed"].as<int>();
        int sizeX = config["x_length"].as<int>();
        int sizeY = config["y_length"].as<int>();
        int sizeZ = config["z_length"].as<int>();
        int type = config["maze_type"].as<int>();
        double scale = 1 / resolution;
        sizeX = sizeX * scale;
        sizeY = sizeY * scale;
        sizeZ = sizeZ * scale;

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
        }
        else {
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

        // ROS
        image_pub_ = nh_.advertise<sensor_msgs::Image>(depth_topic, 1);
        rgb_pub_ = nh_.advertise<sensor_msgs::Image>(rgb_topic, 1);
        image_pub2_ = nh_.advertise<sensor_msgs::Image>(depth_topic2, 1);
        rgb_pub2_ = nh_.advertise<sensor_msgs::Image>(rgb_topic2, 1);
        point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(lidar_topic, 1);
        odom_sub_ = nh_.subscribe(odom_topic, 1, &SensorSimulator::odomCallback, this);
        odom_sub2_ = nh_.subscribe(odom_topic2, 1, &SensorSimulator::odomCallback2, this);
        timer_depth_ = nh_.createTimer(ros::Duration(1 / depth_fps), &SensorSimulator::timerDepthCallback, this);
        timer_depth2_ = nh_.createTimer(ros::Duration(1 / depth_fps), &SensorSimulator::timerDepthCallback2, this);
        timer_lidar_ = nh_.createTimer(ros::Duration(1 / lidar_fps), &SensorSimulator::timerLidarCallback, this);
        timer_map_   = nh_.createTimer(ros::Duration(1), &SensorSimulator::timerMapCallback, this);

        printf("3.Simulation Ready! \n");
        ros::spin();
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void odomCallback2(const nav_msgs::Odometry::ConstPtr &msg);

    void timerDepthCallback(const ros::TimerEvent &);
    void timerDepthCallback2(const ros::TimerEvent &);

    void timerLidarCallback(const ros::TimerEvent &);

    void timerMapCallback(const ros::TimerEvent &);

private:
    bool render_depth{false};
    bool render_lidar{false};
    bool odom_init{false};
    bool odom_init2{false};
    Eigen::Quaternionf quat;
    Eigen::Quaternionf quat_bc, quat_wc;
    Eigen::Vector3f pos;

    Eigen::Quaternionf quat2;
    Eigen::Quaternionf quat_bc2, quat_wc2;
    Eigen::Vector3f pos2;

    CameraParams* camera;
    LidarParams* lidar;
    GridMap* grid_map;
    sensor_msgs::PointCloud2 output;
    
    ros::NodeHandle nh_;
    ros::Publisher image_pub_, rgb_pub_, image_pub2_, rgb_pub2_, point_cloud_pub_;
    ros::Publisher pcl_pub;
    ros::Subscriber odom_sub_, odom_sub2_;
    ros::Timer timer_depth_, timer_depth2_, timer_lidar_, timer_map_;

    // mocka::Maps map;
};



void SensorSimulator::timerDepthCallback(const ros::TimerEvent&) {
    if (!odom_init || !render_depth)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat_wc.w(), quat_wc.x(), quat_wc.y(), quat_wc.z(), pos.x(), pos.y(), pos.z());
    cv::Mat depth_image, rgb_image;
    renderDepthImage(grid_map, camera, T_wc, depth_image, rgb_image);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "生成图像耗时: " << elapsed.count() << " 秒" << std::endl;

    sensor_msgs::Image ros_image;
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = ros::Time::now();
    cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_image.image = depth_image;
    cv_image.toImageMsg(ros_image);
    image_pub_.publish(ros_image);

    cv_bridge::CvImage cv_image_rgb;
    cv_image_rgb.header.stamp = ros::Time::now();
    cv_image_rgb.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
    cv_image_rgb.image = rgb_image;
    cv_image_rgb.toImageMsg(ros_image);
    rgb_pub_.publish(ros_image);
}

void SensorSimulator::timerDepthCallback2(const ros::TimerEvent&) {
    if (!odom_init2 || !render_depth)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat_wc2.w(), quat_wc2.x(), quat_wc2.y(), quat_wc2.z(), pos2.x(), pos2.y(), pos2.z());
    cv::Mat depth_image, rgb_image;
    renderDepthImage(grid_map, camera, T_wc, depth_image, rgb_image);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "生成图像2耗时: " << elapsed.count() << " 秒" << std::endl;

    sensor_msgs::Image ros_image;
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = ros::Time::now();
    cv_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    cv_image.image = depth_image;
    cv_image.toImageMsg(ros_image);
    image_pub2_.publish(ros_image);

    cv_bridge::CvImage cv_image_rgb;
    cv_image_rgb.header.stamp = ros::Time::now();
    cv_image_rgb.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
    cv_image_rgb.image = rgb_image;
    cv_image_rgb.toImageMsg(ros_image);
    rgb_pub2_.publish(ros_image);
}

void SensorSimulator::timerMapCallback(const ros::TimerEvent&) {
    if (pcl_pub.getNumSubscribers() > 0)
        pcl_pub.publish(output);    
}


void SensorSimulator::timerLidarCallback(const ros::TimerEvent&) {
    if (!odom_init || !render_lidar)
        return;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMat::SE3<float> T_wc(quat.w(), quat.x(), quat.y(), quat.z(), pos.x(), pos.y(), pos.z());
    pcl::PointCloud<pcl::PointXYZ> lidar_points;
    renderLidarPointcloud(grid_map, lidar, T_wc, lidar_points);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "生成雷达耗时: " << elapsed.count() << " 秒" << std::endl;

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(lidar_points, output);
    output.header.stamp = ros::Time::now();
    output.header.frame_id = "drone1";
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

    // std::cout << "odom1: " << pos.x() << ", " << pos.y() << ", " << pos.z() << std::endl;
}

void SensorSimulator::odomCallback2(const nav_msgs::Odometry::ConstPtr& msg) {
    quat2.x() = msg->pose.pose.orientation.x;
    quat2.y() = msg->pose.pose.orientation.y;
    quat2.z() = msg->pose.pose.orientation.z;
    quat2.w() = msg->pose.pose.orientation.w;
    quat_wc2 = quat2 * quat_bc2;

    pos2.x() = msg->pose.pose.position.x;
    pos2.y() = msg->pose.pose.position.y;
    pos2.z() = msg->pose.pose.position.z;

    odom_init2 = true;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "sensor_simulator_node");
    ros::NodeHandle nh;

    SensorSimulator sensor_simulator(nh);
    return 0;
}
