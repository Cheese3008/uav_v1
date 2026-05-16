#include "camera_usb_gst_node.hpp"

#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace camera_usb_gst_cpp
{

CameraUsbGstNode::CameraUsbGstNode()
    : Node("camera_usb_gst_node"),
      width_(1280),
      height_(720),
      fps_(30),
      useMjpeg_(true),
      pipeline_(nullptr),
      appSink_(nullptr)
{
    try
    {
        gst_init(nullptr, nullptr);

        loadParameters();
        loadCameraCalibration();

        imagePub_ = this->create_publisher<sensor_msgs::msg::Image>(
            topicName_,
            rclcpp::SensorDataQoS());

        cameraInfoPub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
            cameraInfoTopic_,
            rclcpp::SensorDataQoS());

        startPipeline();

        const auto timerPeriod =
            std::chrono::duration<double>(1.0 / static_cast<double>(fps_));

        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(timerPeriod),
            std::bind(&CameraUsbGstNode::publishFrame, this));

        RCLCPP_INFO(this->get_logger(), "USB Camera GStreamer node started");
        RCLCPP_INFO(this->get_logger(), "Device            : %s", devicePath_.c_str());
        RCLCPP_INFO(this->get_logger(), "Frame ID          : %s", frameId_.c_str());
        RCLCPP_INFO(this->get_logger(), "Image Topic       : %s", topicName_.c_str());
        RCLCPP_INFO(this->get_logger(), "CameraInfo Topic  : %s", cameraInfoTopic_.c_str());
        RCLCPP_INFO(this->get_logger(), "Camera Name       : %s", cameraName_.c_str());
        RCLCPP_INFO(this->get_logger(), "CameraInfo URL    : %s", cameraInfoUrl_.c_str());
        RCLCPP_INFO(this->get_logger(), "Size              : %dx%d @ %d FPS", width_, height_, fps_);
        RCLCPP_INFO(this->get_logger(), "Mode              : %s", useMjpeg_ ? "MJPG" : "YUYV");
    }
    catch (const std::exception &error)
    {
        RCLCPP_ERROR(this->get_logger(), "Khoi tao USB camera node that bai: %s", error.what());
        stopPipeline();
        throw;
    }
}

CameraUsbGstNode::~CameraUsbGstNode()
{
    stopPipeline();
}

void CameraUsbGstNode::loadParameters()
{
    /**
     * Mô tả:
     *     Đọc parameter cấu hình USB camera và file calibration.
     *
     * Input:
     *     ROS2 parameters:
     *         device_path, width, height, fps, frame_id,
     *         topic_name, camera_info_topic, use_mjpeg,
     *         camera_name, camera_info_url.
     *
     * Logic:
     *     Khai báo giá trị mặc định, đọc parameter, kiểm tra dữ liệu hợp lệ.
     *
     * Output:
     *     Cập nhật biến cấu hình nội bộ của node.
     */
    this->declare_parameter<std::string>("device_path", "/dev/camera_usb");
    this->declare_parameter<int>("width", 1280);
    this->declare_parameter<int>("height", 720);
    this->declare_parameter<int>("fps", 30);
    this->declare_parameter<std::string>("frame_id", "camera_usb");
    this->declare_parameter<std::string>("topic_name", "/camera_usb/image_raw");
    this->declare_parameter<std::string>("camera_info_topic", "/camera_usb/camera_info");
    this->declare_parameter<bool>("use_mjpeg", true);

    this->declare_parameter<std::string>("camera_name", "camera_usb");
    this->declare_parameter<std::string>(
        "camera_info_url",
        "file:///home/pihuy/tracktor-beam/camera_info.yaml");

    devicePath_ = this->get_parameter("device_path").as_string();
    width_ = this->get_parameter("width").as_int();
    height_ = this->get_parameter("height").as_int();
    fps_ = this->get_parameter("fps").as_int();
    frameId_ = this->get_parameter("frame_id").as_string();
    topicName_ = this->get_parameter("topic_name").as_string();
    cameraInfoTopic_ = this->get_parameter("camera_info_topic").as_string();
    useMjpeg_ = this->get_parameter("use_mjpeg").as_bool();

    cameraName_ = this->get_parameter("camera_name").as_string();
    cameraInfoUrl_ = normalizeCameraInfoUrl(
        this->get_parameter("camera_info_url").as_string());

    if (devicePath_.empty())
    {
        throw std::runtime_error("device_path dang rong");
    }

    if (topicName_.empty())
    {
        throw std::runtime_error("topic_name dang rong");
    }

    if (cameraInfoTopic_.empty())
    {
        throw std::runtime_error("camera_info_topic dang rong");
    }

    if (frameId_.empty())
    {
        throw std::runtime_error("frame_id dang rong");
    }

    if (cameraName_.empty())
    {
        throw std::runtime_error("camera_name dang rong");
    }

    if (cameraInfoUrl_.empty())
    {
        throw std::runtime_error("camera_info_url dang rong");
    }

    if (width_ <= 0 || height_ <= 0 || fps_ <= 0)
    {
        throw std::runtime_error("width, height hoac fps khong hop le");
    }
}

std::string CameraUsbGstNode::normalizeCameraInfoUrl(
    const std::string &cameraInfoUrl) const
{
    /**
     * Mô tả:
     *     Chuẩn hóa đường dẫn camera_info.yaml.
     *
     * Input:
     *     cameraInfoUrl: đường dẫn từ parameter.
     *
     * Logic:
     *     Nếu URL đã có "://" thì giữ nguyên.
     *     Nếu là đường dẫn thường, ví dụ /home/.../camera_info.yaml,
     *     thì thêm tiền tố file://.
     *
     * Output:
     *     URL hợp lệ cho camera_info_manager.
     */
    if (cameraInfoUrl.empty())
    {
        return "";
    }

    if (cameraInfoUrl.find("://") != std::string::npos)
    {
        return cameraInfoUrl;
    }

    return "file://" + cameraInfoUrl;
}

void CameraUsbGstNode::loadCameraCalibration()
{
    /**
     * Mô tả:
     *     Load file calibration bằng camera_info_manager.
     *
     * Input:
     *     cameraName_, cameraInfoUrl_.
     *
     * Logic:
     *     Tạo CameraInfoManager, load camera_info_url.
     *     Nếu load thất bại thì node vẫn chạy nhưng CameraInfo sẽ là tối thiểu.
     *
     * Output:
     *     cameraInfoManager_ sẵn sàng trả về CameraInfo nếu calibrated.
     */
    cameraInfoManager_ =
        std::make_unique<camera_info_manager::CameraInfoManager>(
            this,
            cameraName_);

    const bool loadOk = cameraInfoManager_->loadCameraInfo(cameraInfoUrl_);

    if (!loadOk)
    {
        RCLCPP_WARN(
            this->get_logger(),
            "Khong load duoc camera_info tu: %s. Node se publish CameraInfo toi thieu.",
            cameraInfoUrl_.c_str());

        return;
    }

    const sensor_msgs::msg::CameraInfo cameraInfoMsg =
        cameraInfoManager_->getCameraInfo();

    RCLCPP_INFO(
        this->get_logger(),
        "Da load camera_info tu: %s",
        cameraInfoUrl_.c_str());

    RCLCPP_INFO(
        this->get_logger(),
        "CameraInfo YAML size: %ux%u",
        cameraInfoMsg.width,
        cameraInfoMsg.height);

    if (cameraInfoMsg.width != static_cast<uint32_t>(width_) ||
        cameraInfoMsg.height != static_cast<uint32_t>(height_))
    {
        RCLCPP_WARN(
            this->get_logger(),
            "Kich thuoc camera_info.yaml (%ux%u) khac voi node config (%dx%d). "
            "Nen calibrate dung do phan giai dang chay de pose/khoang cach chinh xac.",
            cameraInfoMsg.width,
            cameraInfoMsg.height,
            width_,
            height_);
    }
}

std::string CameraUsbGstNode::buildPipelineString() const
{
    /**
     * Mô tả:
     *     Tạo chuỗi pipeline GStreamer đọc USB camera.
     *
     * Input:
     *     devicePath_, width_, height_, fps_, useMjpeg_.
     *
     * Logic:
     *     Nếu use_mjpeg=true thì đọc MJPG, parse JPEG và decode.
     *     Nếu use_mjpeg=false thì đọc raw YUY2.
     *     Thêm queue leaky để bỏ frame cũ khi xử lý không kịp.
     *     Sau đó convert sang BGR để publish encoding bgr8.
     *
     * Output:
     *     Chuỗi pipeline GStreamer kết thúc bằng appsink low-latency.
     */
    std::stringstream pipelineStream;

    pipelineStream
        << "v4l2src device=" << devicePath_
        << " io-mode=2 do-timestamp=true ! ";

    if (useMjpeg_)
    {
        pipelineStream
            << "image/jpeg,width=" << width_
            << ",height=" << height_
            << ",framerate=" << fps_ << "/1 ! "
            << "queue max-size-buffers=1 leaky=downstream ! "
            << "jpegparse ! "
            << "jpegdec ! ";
    }
    else
    {
        pipelineStream
            << "video/x-raw,format=YUY2,width=" << width_
            << ",height=" << height_
            << ",framerate=" << fps_ << "/1 ! "
            << "queue max-size-buffers=1 leaky=downstream ! ";
    }

    pipelineStream
        << "videoconvert ! "
        << "video/x-raw,format=BGR ! "
        << "appsink name=sink emit-signals=false sync=false async=false "
        << "max-buffers=1 drop=true enable-last-sample=false";

    return pipelineStream.str();
}

void CameraUsbGstNode::startPipeline()
{
    /**
     * Mô tả:
     *     Khởi động GStreamer pipeline.
     *
     * Input:
     *     Chuỗi pipeline từ buildPipelineString().
     *
     * Logic:
     *     Parse pipeline, lấy appsink tên sink, set pipeline sang PLAYING.
     *     Không dùng gst_element_get_state() vì live camera có thể trả state không ổn định.
     *
     * Output:
     *     pipeline_ và appSink_ sẵn sàng đọc frame.
     */
    const std::string pipelineString = buildPipelineString();

    RCLCPP_INFO(this->get_logger(), "GStreamer pipeline: %s", pipelineString.c_str());

    GError *gstError = nullptr;

    pipeline_ = gst_parse_launch(pipelineString.c_str(), &gstError);

    if (gstError != nullptr)
    {
        const std::string errorMessage = gstError->message;
        g_error_free(gstError);
        throw std::runtime_error("Loi parse GStreamer pipeline: " + errorMessage);
    }

    if (pipeline_ == nullptr)
    {
        throw std::runtime_error("pipeline_ null sau khi gst_parse_launch");
    }

    appSink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");

    if (appSink_ == nullptr)
    {
        throw std::runtime_error("Khong lay duoc appsink ten sink");
    }

    const GstStateChangeReturn stateResult =
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);

    if (stateResult == GST_STATE_CHANGE_FAILURE)
    {
        throw std::runtime_error("Khong start duoc GStreamer pipeline");
    }

    RCLCPP_INFO(this->get_logger(), "GStreamer pipeline set to PLAYING");
}

void CameraUsbGstNode::stopPipeline()
{
    /**
     * Mô tả:
     *     Dừng pipeline và giải phóng camera.
     *
     * Input:
     *     pipeline_, appSink_ hiện tại.
     *
     * Logic:
     *     Set pipeline về GST_STATE_NULL, sau đó unref object.
     *
     * Output:
     *     Camera được release khi node dừng.
     */
    if (pipeline_ != nullptr)
    {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
    }

    if (appSink_ != nullptr)
    {
        gst_object_unref(appSink_);
        appSink_ = nullptr;
    }

    if (pipeline_ != nullptr)
    {
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
}

void CameraUsbGstNode::publishFrame()
{
    /**
     * Mô tả:
     *     Đọc frame từ appsink và publish Image + CameraInfo.
     *
     * Input:
     *     Frame BGR từ GStreamer appsink.
     *
     * Logic:
     *     Pull sample với timeout ngắn để tránh block lâu.
     *     Lấy buffer/caps, map dữ liệu ảnh, tạo Image message.
     *     CameraInfo được lấy từ camera_info.yaml thông qua camera_info_manager.
     *
     * Output:
     *     Publish ảnh và camera info đồng bộ header.
     */
    if (appSink_ == nullptr)
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "appsink_ null, khong the doc frame");
        return;
    }

    GstSample *sample = gst_app_sink_try_pull_sample(
        GST_APP_SINK(appSink_),
        5 * GST_MSECOND);

    if (sample == nullptr)
    {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "Khong lay duoc frame tu USB camera");
        return;
    }

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstCaps *caps = gst_sample_get_caps(sample);

    if (buffer == nullptr || caps == nullptr)
    {
        gst_sample_unref(sample);

        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "Sample khong co buffer hoac caps");
        return;
    }

    GstStructure *structure = gst_caps_get_structure(caps, 0);

    int imageWidth = 0;
    int imageHeight = 0;

    const gboolean hasWidth = gst_structure_get_int(structure, "width", &imageWidth);
    const gboolean hasHeight = gst_structure_get_int(structure, "height", &imageHeight);

    if (!hasWidth || !hasHeight)
    {
        gst_sample_unref(sample);

        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "Khong lay duoc width/height tu GstCaps");
        return;
    }

    GstMapInfo mapInfo;

    if (!gst_buffer_map(buffer, &mapInfo, GST_MAP_READ))
    {
        gst_sample_unref(sample);

        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "Khong map duoc GstBuffer");
        return;
    }

    try
    {
        sensor_msgs::msg::Image imageMsg = createImageMessage(
            mapInfo.data,
            mapInfo.size,
            imageWidth,
            imageHeight);

        sensor_msgs::msg::CameraInfo cameraInfoMsg =
            createCameraInfoMessage(imageMsg.header);

        imagePub_->publish(imageMsg);
        cameraInfoPub_->publish(cameraInfoMsg);
    }
    catch (const std::exception &error)
    {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "Loi publish frame: %s",
            error.what());
    }

    gst_buffer_unmap(buffer, &mapInfo);
    gst_sample_unref(sample);
}

sensor_msgs::msg::Image CameraUsbGstNode::createImageMessage(
    const guint8 *dataPtr,
    const gsize dataSize,
    const int imageWidth,
    const int imageHeight)
{
    /**
     * Mô tả:
     *     Tạo sensor_msgs::msg::Image từ buffer BGR.
     *
     * Input:
     *     dataPtr, dataSize, imageWidth, imageHeight.
     *
     * Logic:
     *     Kiểm tra kích thước buffer, gán header, encoding bgr8,
     *     copy dữ liệu ảnh vào message.
     *
     * Output:
     *     Image message hợp lệ.
     */
    if (dataPtr == nullptr)
    {
        throw std::runtime_error("dataPtr null");
    }

    if (imageWidth <= 0 || imageHeight <= 0)
    {
        throw std::runtime_error("kich thuoc anh khong hop le");
    }

    const size_t expectedSize =
        static_cast<size_t>(imageWidth) *
        static_cast<size_t>(imageHeight) *
        3U;

    if (dataSize < expectedSize)
    {
        std::stringstream errorStream;

        errorStream
            << "buffer anh nho hon kich thuoc BGR mong doi, dataSize="
            << dataSize
            << ", expectedSize="
            << expectedSize;

        throw std::runtime_error(errorStream.str());
    }

    sensor_msgs::msg::Image imageMsg;

    imageMsg.header.stamp = this->get_clock()->now();
    imageMsg.header.frame_id = frameId_;

    imageMsg.height = static_cast<uint32_t>(imageHeight);
    imageMsg.width = static_cast<uint32_t>(imageWidth);
    imageMsg.encoding = "bgr8";
    imageMsg.is_bigendian = false;
    imageMsg.step = static_cast<uint32_t>(imageWidth * 3);

    imageMsg.data.resize(expectedSize);
    std::memcpy(imageMsg.data.data(), dataPtr, expectedSize);

    return imageMsg;
}

sensor_msgs::msg::CameraInfo CameraUsbGstNode::createCameraInfoMessage(
    const std_msgs::msg::Header &imageHeader) const
{
    /**
     * Mô tả:
     *     Tạo CameraInfo message từ file calibration YAML.
     *
     * Input:
     *     imageHeader: header của Image message.
     *
     * Logic:
     *     Nếu cameraInfoManager_ đã load được camera_info.yaml thì lấy trực tiếp
     *     thông số D, K, R, P từ file calib.
     *     Nếu chưa load được file calib thì publish CameraInfo tối thiểu.
     *     Header được cập nhật theo Image để đồng bộ timestamp và frame_id.
     *
     * Output:
     *     CameraInfo message publish ra topic camera info đã cấu hình.
     */
    sensor_msgs::msg::CameraInfo cameraInfoMsg;

    if (cameraInfoManager_ != nullptr && cameraInfoManager_->isCalibrated())
    {
        cameraInfoMsg = cameraInfoManager_->getCameraInfo();
    }
    else
    {
        cameraInfoMsg.width = static_cast<uint32_t>(width_);
        cameraInfoMsg.height = static_cast<uint32_t>(height_);
        cameraInfoMsg.distortion_model = "plumb_bob";
    }

    cameraInfoMsg.header = imageHeader;

    return cameraInfoMsg;
}

}  // namespace camera_usb_gst_cpp

int main(int argc, char **argv)
{
    /**
     * Mô tả:
     *     Hàm main chạy ROS2 node đọc USB camera bằng GStreamer.
     *
     * Input:
     *     argc, argv từ dòng lệnh ROS2.
     *
     * Logic:
     *     Khởi tạo rclcpp, tạo CameraUsbGstNode,
     *     spin node để publish ảnh và camera info liên tục,
     *     shutdown an toàn khi node dừng.
     *
     * Output:
     *     Node publish ảnh USB camera ra topic đã cấu hình.
     */
    rclcpp::init(argc, argv);

    try
    {
        auto node = std::make_shared<camera_usb_gst_cpp::CameraUsbGstNode>();
        rclcpp::spin(node);
    }
    catch (const std::exception &error)
    {
        std::cerr << "Loi camera_usb_gst_node: " << error.what() << std::endl;
    }

    rclcpp::shutdown();

    return 0;
}