#ifndef CAMERA_USB_GST_CPP_CAMERA_USB_GST_NODE_HPP_
#define CAMERA_USB_GST_CPP_CAMERA_USB_GST_NODE_HPP_

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/header.hpp>

#include <string>

namespace camera_usb_gst_cpp
{

class CameraUsbGstNode : public rclcpp::Node
{
public:
    explicit CameraUsbGstNode();
    ~CameraUsbGstNode() override;

private:
    /**
     * Mô tả:
     *     Đọc toàn bộ parameter cấu hình USB camera, topic ảnh và topic CameraInfo.
     *
     * Input:
     *     Không có input trực tiếp.
     *     Dữ liệu lấy từ ROS2 parameter:
     *         device_path, width, height, fps, frame_id,
     *         topic_name, camera_info_topic, use_mjpeg.
     *
     * Logic:
     *     Khai báo parameter mặc định, đọc giá trị parameter,
     *     kiểm tra các giá trị quan trọng để tránh chạy sai cấu hình.
     *
     * Output:
     *     Cập nhật các biến cấu hình nội bộ của node.
     */
    void loadParameters();

    /**
     * Mô tả:
     *     Tạo chuỗi pipeline GStreamer để đọc USB camera.
     *
     * Input:
     *     devicePath_ : đường dẫn camera, ví dụ /dev/camera_front hoặc /dev/camera_down.
     *     width_      : chiều rộng ảnh.
     *     height_     : chiều cao ảnh.
     *     fps_        : tốc độ khung hình.
     *     useMjpeg_   : chọn MJPG hoặc raw YUYV.
     *
     * Logic:
     *     Với USB camera hiện tại, MJPG cần io-mode=2 và jpegparse
     *     để tránh lỗi not-negotiated.
     *     Pipeline chuyển ảnh cuối cùng sang BGR để publish encoding bgr8.
     *
     * Output:
     *     Trả về chuỗi pipeline GStreamer hoàn chỉnh.
     */
    std::string buildPipelineString() const;

    /**
     * Mô tả:
     *     Khởi động pipeline GStreamer.
     *
     * Input:
     *     Pipeline string được tạo từ buildPipelineString().
     *
     * Logic:
     *     Parse pipeline bằng gst_parse_launch,
     *     lấy appsink tên "sink",
     *     chuyển pipeline sang trạng thái PLAYING.
     *
     * Output:
     *     pipeline_ và appSink_ sẵn sàng để đọc frame.
     */
    void startPipeline();

    /**
     * Mô tả:
     *     Dừng pipeline GStreamer và giải phóng USB camera.
     *
     * Input:
     *     pipeline_ và appSink_ hiện tại.
     *
     * Logic:
     *     Chuyển pipeline về GST_STATE_NULL,
     *     sau đó unref appSink_ và pipeline_.
     *
     * Output:
     *     Camera được release sạch khi node tắt.
     */
    void stopPipeline();

    /**
     * Mô tả:
     *     Callback timer chính để lấy frame từ appsink và publish ra ROS2.
     *
     * Input:
     *     Frame mới nhất từ GStreamer appsink.
     *
     * Logic:
     *     Pull sample từ appsink,
     *     lấy buffer và caps,
     *     map dữ liệu ảnh,
     *     tạo sensor_msgs::msg::Image,
     *     tạo sensor_msgs::msg::CameraInfo cùng timestamp,
     *     publish ra topic ảnh và topic camera info đã cấu hình.
     *
     * Output:
     *     Publish ảnh BGR8 và CameraInfo ra ROS2 topic.
     */
    void publishFrame();

    /**
     * Mô tả:
     *     Tạo ROS2 Image message từ buffer ảnh BGR.
     *
     * Input:
     *     dataPtr     : con trỏ dữ liệu ảnh BGR.
     *     dataSize    : kích thước buffer ảnh.
     *     imageWidth  : chiều rộng ảnh.
     *     imageHeight : chiều cao ảnh.
     *
     * Logic:
     *     Kiểm tra buffer hợp lệ,
     *     gán header stamp, frame_id,
     *     encoding bgr8, step, width, height,
     *     sau đó copy dữ liệu ảnh vào message.
     *
     * Output:
     *     Trả về sensor_msgs::msg::Image hợp lệ.
     */
    sensor_msgs::msg::Image createImageMessage(
        const guint8 *dataPtr,
        const gsize dataSize,
        const int imageWidth,
        const int imageHeight);

    /**
     * Mô tả:
     *     Tạo ROS2 CameraInfo message từ thông số calibration hiện tại.
     *
     * Input:
     *     imageHeader: header của ảnh hiện tại để đồng bộ timestamp/frame_id.
     *
     * Logic:
     *     Gán width, height, distortion_model, camera_matrix K,
     *     distortion coefficients D, rectification matrix R và projection matrix P.
     *     Hiện tại thông số này đang dùng calibration mặc định 1280x720.
     *
     * Output:
     *     Trả về sensor_msgs::msg::CameraInfo để publish ra topic camera info.
     */
    sensor_msgs::msg::CameraInfo createCameraInfoMessage(
        const std_msgs::msg::Header &imageHeader) const;

private:
    std::string devicePath_;
    std::string frameId_;
    std::string topicName_;
    std::string cameraInfoTopic_;

    int width_;
    int height_;
    int fps_;

    bool useMjpeg_;

    GstElement *pipeline_;
    GstElement *appSink_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imagePub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfoPub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace camera_usb_gst_cpp

#endif  // CAMERA_USB_GST_CPP_CAMERA_USB_GST_NODE_HPP_