#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <opencv2/opencv.hpp>

using namespace std::chrono_literals;

class UdpH264CameraNode : public rclcpp::Node
{
public:
  UdpH264CameraNode()
  : Node("udp_h264_camera_node")
  {
    declare_parameter<std::string>("frame_id", "camera_link");
    declare_parameter<std::string>("image_topic", "/camera/image");
    declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");

    declare_parameter<int>("udp_port", 5600);
    declare_parameter<int>("width", 1280);
    declare_parameter<int>("height", 960);
    declare_parameter<double>("fps", 30.0);

    declare_parameter<double>("fx", 542.6);
    declare_parameter<double>("fy", 542.6);
    declare_parameter<double>("cx", 640.0);
    declare_parameter<double>("cy", 480.0);

    declare_parameter<std::vector<double>>(
      "distortion_coeffs",
      std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0});

    declare_parameter<bool>("use_custom_pipeline", false);
    declare_parameter<std::string>("custom_pipeline", "");

    get_parameter("frame_id", frame_id_);
    get_parameter("image_topic", image_topic_);
    get_parameter("camera_info_topic", camera_info_topic_);
    get_parameter("udp_port", udp_port_);
    get_parameter("width", width_);
    get_parameter("height", height_);
    get_parameter("fps", fps_);
    get_parameter("fx", fx_);
    get_parameter("fy", fy_);
    get_parameter("cx", cx_);
    get_parameter("cy", cy_);
    get_parameter("distortion_coeffs", distortion_coeffs_);
    get_parameter("use_custom_pipeline", use_custom_pipeline_);
    get_parameter("custom_pipeline", custom_pipeline_);

    auto qos = rclcpp::SensorDataQoS();

    image_pub_ = create_publisher<sensor_msgs::msg::Image>(image_topic_, qos);
    camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(camera_info_topic_, qos);

    camera_info_msg_ = makeCameraInfo();

    openStream();

    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, fps_));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(period),
      std::bind(&UdpH264CameraNode::captureLoop, this));

    RCLCPP_INFO(get_logger(), "UDP H264 camera node started");
    RCLCPP_INFO(get_logger(), "image_topic      : %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "camera_info_topic: %s", camera_info_topic_.c_str());
    RCLCPP_INFO(get_logger(), "frame_id         : %s", frame_id_.c_str());
    RCLCPP_INFO(get_logger(), "udp_port         : %d", udp_port_);
  }

  ~UdpH264CameraNode() override
  {
    if (cap_.isOpened()) {
      cap_.release();
    }
  }

private:
  sensor_msgs::msg::CameraInfo makeCameraInfo()
  {
    sensor_msgs::msg::CameraInfo msg;
    msg.header.frame_id = frame_id_;
    msg.width = static_cast<uint32_t>(width_);
    msg.height = static_cast<uint32_t>(height_);
    msg.distortion_model = "plumb_bob";
    msg.d = distortion_coeffs_;

    msg.k = {
      fx_, 0.0, cx_,
      0.0, fy_, cy_,
      0.0, 0.0, 1.0
    };

    msg.r = {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0
    };

    msg.p = {
      fx_, 0.0, cx_, 0.0,
      0.0, fy_, cy_, 0.0,
      0.0, 0.0, 1.0, 0.0
    };

    return msg;
  }

  std::string defaultPipeline() const
  {
    return
      "udpsrc port=" + std::to_string(udp_port_) +
      " caps=\"application/x-rtp, media=video, encoding-name=H264, payload=96\" ! "
      "rtpjitterbuffer latency=50 ! "
      "rtph264depay ! "
      "h264parse ! "
      "avdec_h264 ! "
      "videoconvert ! "
      "video/x-raw,format=BGR ! "
      "appsink sync=false drop=true max-buffers=1";
  }

  void openStream()
  {
    const std::string pipeline =
      (use_custom_pipeline_ && !custom_pipeline_.empty()) ? custom_pipeline_ : defaultPipeline();

    RCLCPP_INFO(get_logger(), "Opening pipeline: %s", pipeline.c_str());

    cap_.open(pipeline, cv::CAP_GSTREAMER);

    if (!cap_.isOpened()) {
      RCLCPP_ERROR(get_logger(), "Failed to open UDP H264 stream");
    }
  }

  void publishCameraInfo(const rclcpp::Time &stamp)
  {
    camera_info_msg_.header.stamp = stamp;
    camera_info_pub_->publish(camera_info_msg_);
  }

  void captureLoop()
  {
    if (!cap_.isOpened()) {
      static int retry_count = 0;
      if ((retry_count++ % 30) == 0) {
        RCLCPP_WARN(get_logger(), "Stream not opened, retrying...");
      }
      openStream();
      return;
    }

    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "No frame received from UDP stream");
      return;
    }

    const auto stamp = now();

    if (frame.cols != width_ || frame.rows != height_) {
      width_ = frame.cols;
      height_ = frame.rows;
      camera_info_msg_ = makeCameraInfo();
    }

    cv_bridge::CvImage cv_img;
    cv_img.header.stamp = stamp;
    cv_img.header.frame_id = frame_id_;
    cv_img.encoding = sensor_msgs::image_encodings::BGR8;
    cv_img.image = frame;

    image_pub_->publish(*cv_img.toImageMsg());
    publishCameraInfo(stamp);
  }

private:
  std::string frame_id_;
  std::string image_topic_;
  std::string camera_info_topic_;

  int udp_port_{5600};
  int width_{1280};
  int height_{720};
  double fps_{30.0};

  double fx_{640.0};
  double fy_{640.0};
  double cx_{640.0};
  double cy_{360.0};

  std::vector<double> distortion_coeffs_;

  bool use_custom_pipeline_{false};
  std::string custom_pipeline_;

  cv::VideoCapture cap_;
  sensor_msgs::msg::CameraInfo camera_info_msg_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<UdpH264CameraNode>());
  rclcpp::shutdown();
  return 0;
}