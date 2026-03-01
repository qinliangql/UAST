#include "sensor_simulator.cuh"

namespace raycast
{   
    GridMap::GridMap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float resolution, int occupy_threshold = 1){
        
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        float length = max_pt(0) - min_pt(0);   // X方向的长度
        float width  = max_pt(1) - min_pt(1);   // Y方向的宽度
        float height = max_pt(2) - min_pt(2);   // Z方向的高度
        Vector3f origin(min_pt(0), min_pt(1), min_pt(2));
        Vector3f map_size(length, width, height);
        origin_x_ = origin.x;
        origin_y_ = origin.y;
        origin_z_ = origin.z;

        Vector3i grid_size;
        grid_size.x = ceil(map_size.x / resolution);
        grid_size.y = ceil(map_size.y / resolution);
        grid_size.z = ceil(map_size.z / resolution);
        int grid_total_size = grid_size.x * grid_size.y * grid_size.z;

        resolution_   = resolution;
        grid_size_x_  = grid_size.x, 
        grid_size_y_  = grid_size.y, 
        grid_size_z_  = grid_size.z, 
        grid_size_yz_ = grid_size.y * grid_size.z;
        occupy_threshold_ = occupy_threshold;
        raycast_step_ = resolution;

        std::vector<int> h_map(grid_total_size, 0);
        // 有时候会有全空的行，加个很小的偏移
        for (size_t i = 0; i < cloud->points.size(); i++) {
            Vector3f point(cloud->points[i].x + 0.001, cloud->points[i].y + 0.001, cloud->points[i].z + 0.001);
            int idx = Vox2Idx(Pos2Vox(point));
            if (idx < grid_total_size) {
                h_map[idx]++;
            }
        }
        cudaMalloc((void **)&map_cuda_, grid_total_size * sizeof(int));
        cudaMemcpy(map_cuda_, h_map.data(), grid_total_size * sizeof(int), cudaMemcpyHostToDevice);
    }

    __host__ __device__ Vector3i GridMap::Pos2Vox(const Vector3f &pos)
    {
        Vector3i vox;
        vox.x = floor((pos.x - origin_x_) / resolution_);
        vox.y = floor((pos.y - origin_y_) / resolution_);
        vox.z = floor((pos.z - origin_z_) / resolution_);
        return vox;
    }

    __host__ __device__ Vector3f GridMap::Vox2Pos(const Vector3i &vox)
    {
        Vector3f pos;
        pos.x = (vox.x + 0.5f) * resolution_ + origin_x_;
        pos.y = (vox.y + 0.5f) * resolution_ + origin_y_;
        pos.z = (vox.z + 0.5f) * resolution_ + origin_z_;
        return pos;
    }

    __host__ __device__ int GridMap::Vox2Idx(const Vector3i &vox)
    {
        return vox.x * grid_size_yz_ + vox.y * grid_size_z_ + vox.z;
    }

    __host__ __device__ Vector3i GridMap::Idx2Vox(int idx)
    {
        return Vector3i(idx / grid_size_yz_, (idx % grid_size_yz_) / grid_size_z_, idx % grid_size_z_);
    }

    __device__ int GridMap::symmetricIndex(int index, int length)
    {
        index = index % (2 * length - 2);
        if (index < 0)
        {
            index += (2 * length - 2);
        }

        if (index >= length)
        {
            index = 2 * length - 2 - index;
        }
        return index;
    }

    // -1: z越界; 0: 空闲; 1: 占据
    __device__  int GridMap::mapQuery(const Vector3f &pos){
        Vector3i vox = Pos2Vox(pos);
        vox.x = symmetricIndex(vox.x, grid_size_x_);
        vox.y = symmetricIndex(vox.y, grid_size_y_);

        if (vox.z >= grid_size_z_)
            return 0;
        if (vox.z <= 0)
            return 1;

        int idx = Vox2Idx(vox);
        if (map_cuda_[idx] > occupy_threshold_)
            return 1;
        return 0;        
    }

    __global__ void cameraRaycastKernel(float* depth_values, float* z_heights, GridMap grid_map, CameraParams camera_param, cudaMat::SE3<float> T_wc)
    {
        int u = threadIdx.x;
        int v = blockIdx.x;

        // printf("u: %d, v: %d \n", u, v);

        if (u < camera_param.image_width && v < camera_param.image_height)
        {
            // 计算射线方向
            float y = -(u - camera_param.cx) / camera_param.fx;
            float z = -(v - camera_param.cy) / camera_param.fy;
            float x = 1.0f;

            // 归一化射线方向
            float length = sqrtf(x * x + y * y + z * z);
            x /= length;
            y /= length;
            z /= length;

            // 计算每个轴的增量比例 (x方向固定步长避免近距离处畸变; 0.5是瞎设的防止过于稀疏导致错误)
            float dx = 0.5 * grid_map.raycast_step_;
            float dy = (y / x) * dx;
            float dz = (z / x) * dx;

            // 递增射线方向上的每个轴
            int scale = 0;
            float depth = 0.0f;
            float z_height = 0.0f;  // 新增：存储世界系z坐标

            while (1)
            {
                scale += 1;

                float point_x = scale * dx;
                float point_y = scale * dy;
                float point_z = scale * dz;

                float3 point_c = make_float3(point_x, point_y, point_z);
                float3 point_w = T_wc * point_c;

                Vector3f point(point_w.x, point_w.y, point_w.z);

                int occupied = grid_map.mapQuery(point);

                if (occupied == 1)
                {
                    // depth = point_x;  // 直接这样赋值会有一点误差
                    // 栅格化避免平面变曲面 (有些冗余，但在机体系栅格化会有类似摩尔纹的东西)
                    Vector3i occ_vox_w = grid_map.Pos2Vox(point);
                    Vector3f occ_point_w = grid_map.Vox2Pos(occ_vox_w);
                    float3 occ_point_w_ = make_float3(occ_point_w.x, occ_point_w.y, occ_point_w.z);
                    float3 occ_point_c_ = T_wc.inv() * occ_point_w_;
                    depth = occ_point_c_.x;
                    z_height = occ_point_w.z;
                    break;
                }

                if (point_x >= camera_param.max_depth_dist){
                    depth = camera_param.max_depth_dist;
                    // 设置一个特殊的z_height值来标识超出范围区域
                    z_height = -1.0f;  // 使用-1作为超出范围的标识
                    break;
                }
            }

            // 将深度值存储到输出数组中
            if (camera_param.normalize_depth)
                depth = depth / camera_param.max_depth_dist;
            depth_values[v * camera_param.image_width + u] = depth;
            z_heights[v * camera_param.image_width + u] = z_height;
        }
    }

    void renderDepthImage(GridMap* grid_map, CameraParams* camera_param, cudaMat::SE3<float>& T_wc, cv::Mat& depth_image, cv::Mat& rgb_image)
    {   
        float* depth_values;
        float* z_heights;
        size_t num_elements = camera_param->image_width * camera_param->image_height;
        cudaMallocManaged(&depth_values, num_elements * sizeof(float));
        cudaMallocManaged(&z_heights, num_elements * sizeof(float));

        // 在GPU上启动核函数
        cameraRaycastKernel<<<camera_param->image_height, camera_param->image_width>>>(depth_values, z_heights, *grid_map, *camera_param, T_wc);
        
        cudaDeviceSynchronize();

        depth_image.create(camera_param->image_height, camera_param->image_width, CV_32FC1);

        cudaMemcpy(depth_image.data, depth_values, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        // 4. 生成RGB热力图（基于z_heights）
        rgb_image.create(camera_param->image_height, camera_param->image_width, CV_8UC3);  // 8位3通道
        // 4.1 归一化z高度到[0, 255]
        float z_min = 0.0;  // 最小高度
        float z_max = 10;  // 最大高度
        float z_range = z_max - z_min;
        if (z_range < 1e-6) z_range = 1e-6;  // 避免除零

        // 定义地面高度阈值（根据实际场景调整）
        float ground_threshold = 0.1;  // 地面高度阈值，可根据实际情况调整
        // 新的地面颜色：深绿色（BGR格式）
        int ground_b = 0;    // 蓝色通道
        int ground_g = 128;  // 绿色通道
        int ground_r = 0;    // 红色通道
        
        // 定义超出范围区域的颜色：灰色
        int out_of_range_b = 128;  // 蓝色通道
        int out_of_range_g = 128;  // 绿色通道
        int out_of_range_r = 128;  // 红色通道
        
        // 遍历每个像素，映射z高度到RGB
        for (int v = 0; v < camera_param->image_height; v++) {
            for (int u = 0; u < camera_param->image_width; u++) {
                int idx = v * camera_param->image_width + u;
                float z = z_heights[idx];
                
                // 检查是否为超出范围区域
                if (z < -0.5f) {  // 如果z_height小于-0.5，视为超出范围
                    // 设置为灰色
                    rgb_image.at<cv::Vec3b>(v, u)[0] = out_of_range_b;
                    rgb_image.at<cv::Vec3b>(v, u)[1] = out_of_range_g;
                    rgb_image.at<cv::Vec3b>(v, u)[2] = out_of_range_r;
                }
                // 检查是否为地面像素
                else if (z >= z_min && z <= ground_threshold) {
                    // 设置为深绿色
                    rgb_image.at<cv::Vec3b>(v, u)[0] = ground_b;
                    rgb_image.at<cv::Vec3b>(v, u)[1] = ground_g;
                    rgb_image.at<cv::Vec3b>(v, u)[2] = ground_r;
                }
                else {
                    // 对非地面且在范围内的像素使用jet配色
                    float norm_z = (z - z_min) / z_range;
                    
                    // jet配色：蓝(0)→青→绿→黄→红(1)
                    float r = std::min(std::max(1.5f - std::abs(4*norm_z - 3), 0.0f), 1.0f);
                    float g = std::min(std::max(1.5f - std::abs(4*norm_z - 2), 0.0f), 1.0f);
                    float b = std::min(std::max(1.5f - std::abs(4*norm_z - 1), 0.0f), 1.0f);
                    
                    // 转换为8位BGR
                    rgb_image.at<cv::Vec3b>(v, u)[0] = static_cast<uchar>(b * 255);
                    rgb_image.at<cv::Vec3b>(v, u)[1] = static_cast<uchar>(g * 255);
                    rgb_image.at<cv::Vec3b>(v, u)[2] = static_cast<uchar>(r * 255);
                }
            }
        }

        cudaFree(depth_values);
        cudaFree(z_heights);  // 释放z高度内存
        return;
    }

    __global__ void lidarRaycastKernel(Vector3f* point_values, GridMap grid_map, LidarParams lidar_param, cudaMat::SE3<float> T_wc)
    {
        int h = threadIdx.x;
        int v = blockIdx.x;

        // printf("u: %d, v: %d \n", u, v);
        if (h < lidar_param.horizontal_num && v < lidar_param.vertical_lines)
        {   
            float vertical_resolution = (lidar_param.vertical_angle_end - lidar_param.vertical_angle_start) / (lidar_param.vertical_lines - 1);
            float vertical_angle = lidar_param.vertical_angle_start + v * vertical_resolution;
            float sin_vert = std::sin(vertical_angle * M_PI / 180.0);
            float cos_vert = std::cos(vertical_angle * M_PI / 180.0);
            float horizontal_angle = h * lidar_param.horizontal_resolution;
            float sin_horz = std::sin(horizontal_angle * M_PI / 180.0);
            float cos_horz = std::cos(horizontal_angle * M_PI / 180.0);
            // 计算射线方向
            Vector3f ray_direction(cos_vert * cos_horz, cos_vert * sin_horz, sin_vert);

            // 计算每个轴的增量比例
            float dx = ray_direction.x * grid_map.raycast_step_;
            float dy = ray_direction.y * grid_map.raycast_step_;
            float dz = ray_direction.z * grid_map.raycast_step_;

            // 递增射线方向上的每个轴
            int scale = 0;
            Vector3f point_value(0, 0, 0);

            while (1)
            {
                scale += 1;

                float point_x = scale * dx;
                float point_y = scale * dy;
                float point_z = scale * dz;

                float3 point_c = make_float3(point_x, point_y, point_z);
                float3 point_w = T_wc * point_c;

                Vector3f point(point_w.x, point_w.y, point_w.z);

                int occupied = grid_map.mapQuery(point);

                float ray_length = sqrtf(point_x * point_x + point_y * point_y + point_z * point_z);

                if (occupied == 1)
                {
                    point_value = Vector3f(point_x, point_y, point_z);
                    Vector3i vox_body = grid_map.Pos2Vox(point_value);  // 栅格化避免平面变曲面
                    point_value = grid_map.Vox2Pos(vox_body);
                    break;
                }

                if (ray_length > lidar_param.max_lidar_dist){
                    break;
                }
            }

            // 将点云值存储到输出数组中，(0, 0, 0)为无效值
            point_values[v * lidar_param.horizontal_num + h] = point_value;
        }
    }

    void renderLidarPointcloud(GridMap *grid_map, LidarParams *lidar_param, cudaMat::SE3<float>& T_wc, pcl::PointCloud<pcl::PointXYZ>& lidar_points){
        Vector3f* point_values;
        size_t num_elements = lidar_param->vertical_lines * lidar_param->horizontal_num;
        cudaMallocManaged(&point_values, num_elements * sizeof(Vector3f));

        // 在GPU上启动核函数
        lidarRaycastKernel<<<lidar_param->vertical_lines, lidar_param->horizontal_num>>>(point_values, *grid_map, *lidar_param, T_wc);
        
        cudaDeviceSynchronize();

        std::vector<Vector3f> cpu_points(num_elements);
        cudaMemcpy(cpu_points.data(), point_values, num_elements * sizeof(Vector3f), cudaMemcpyDeviceToHost);
        
        lidar_points.points.clear();
        lidar_points.points.reserve(num_elements);
        
        for (const auto& point : cpu_points) {
            if (point.x != 0 || point.y != 0 || point.z != 0) {
                lidar_points.points.emplace_back(point.x, point.y, point.z);
            }
        }
        cudaFree(point_values);
        return;
    }


    
}