#ifndef CAMERA_USB_GST_NODE_HPP_
#define CAMERA_USB_GST_NODE_HPP_

#include <memory>
#include <string>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/header.hpp>

#include <camera_info_manager/camera_info_manager.hpp>

namespace camera_usb_gst_cpp
{

class CameraUsbGstNode : public rclcpp::Node
{
public:
    /**
     * Mô tả:
     *     Constructor khởi tạo node đọc USB camera bằng GStreamer.
     *
     * Input:
     *     Không có input trực tiếp.
     *
     * Logic:
     *     Đọc parameter, load camera_info.yaml, tạo publisher,
     *     khởi động pipeline GStreamer và timer publish ảnh.
     *
     * Output:
     *     Node sẵn sàng publish Image và CameraInfo.
     */
    CameraUsbGstNode();

    /**
     * Mô tả:
     *     Destructor dừng pipeline và giải phóng tài nguyên camera.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     Gọi stopPipeline().
     *
     * Output:
     *     Camera được release an toàn.
     */
    ~CameraUsbGstNode();

private:
    /**
     * Mô tả:
     *     Đọc parameter cấu hình USB camera và camera calibration.
     *
     * Input:
     *     ROS2 parameters.
     *
     * Logic:
     *     Khai báo parameter mặc định, đọc giá trị, kiểm tra hợp lệ.
     *
     * Output:
     *     Cập nhật biến cấu hình nội bộ.
     */
    void loadParameters();

    /**
     * Mô tả:
     *     Chuẩn hóa đường dẫn camera_info.
     *
     * Input:
     *     cameraInfoUrl: có thể là đường dẫn thường hoặc file URI.
     *
     * Logic:
     *     Nếu đã có dạng xxx:// thì giữ nguyên.
     *     Nếu là đường dẫn thường thì thêm file:// phía trước.
     *
     * Output:
     *     URL hợp lệ cho camera_info_manager.
     */
    std::string normalizeCameraInfoUrl(const std::string &cameraInfoUrl) const;

    /**
     * Mô tả:
     *     Load thông số camera calibration từ file YAML.
     *
     * Input:
     *     cameraName_, cameraInfoUrl_.
     *
     * Logic:
     *     Tạo CameraInfoManager và load file calibration.
     *     Nếu không load được thì cảnh báo và publish CameraInfo tối thiểu.
     *
     * Output:
     *     cameraInfoManager_ chứa thông số calibration nếu load thành công.
     */
    void loadCameraCalibration();

    /**
     * Mô tả:
     *     Tạo chuỗi pipeline GStreamer.
     *
     * Input:
     *     devicePath_, width_, height_, fps_, useMjpeg_.
     *
     * Logic:
     *     Dùng MJPEG hoặc YUY2, thêm queue leaky để giảm delay,
     *     convert sang BGR và đẩy ra appsink.
     *
     * Output:
     *     Chuỗi pipeline GStreamer.
     */
    std::string buildPipelineString() const;

    /**
     * Mô tả:
     *     Khởi động pipeline GStreamer.
     *
     * Input:
     *     Pipeline string từ buildPipelineString().
     *
     * Logic:
     *     Parse pipeline, lấy appsink, set pipeline sang PLAYING.
     *
     * Output:
     *     pipeline_ và appSink_ sẵn sàng đọc frame.
     */
    void startPipeline();

    /**
     * Mô tả:
     *     Dừng pipeline GStreamer.
     *
     * Input:
     *     pipeline_, appSink_ hiện tại.
     *
     * Logic:
     *     Set pipeline về NULL và unref object.
     *
     * Output:
     *     Camera được giải phóng.
     */
    void stopPipeline();

    /**
     * Mô tả:
     *     Đọc frame từ appsink và publish Image + CameraInfo.
     *
     * Input:
     *     Frame từ GStreamer appsink.
     *
     * Logic:
     *     Pull sample, map buffer, tạo Image message,
     *     lấy CameraInfo từ camera_info_manager rồi publish.
     *
     * Output:
     *     Publish Image và CameraInfo đồng bộ timestamp.
     */
    void publishFrame();

    /**
     * Mô tả:
     *     Tạo sensor_msgs::msg::Image từ buffer BGR.
     *
     * Input:
     *     dataPtr: con trỏ dữ liệu ảnh.
     *     dataSize: kích thước buffer.
     *     imageWidth: chiều rộng ảnh.
     *     imageHeight: chiều cao ảnh.
     *
     * Logic:
     *     Kiểm tra buffer, set header, encoding bgr8 và copy dữ liệu.
     *
     * Output:
     *     Image message hợp lệ.
     */
    sensor_msgs::msg::Image createImageMessage(
        const guint8 *dataPtr,
        const gsize dataSize,
        const int imageWidth,
        const int imageHeight);

    /**
     * Mô tả:
     *     Tạo CameraInfo message từ file calibration YAML.
     *
     * Input:
     *     imageHeader: header của Image message.
     *
     * Logic:
     *     Nếu đã load calibration thì lấy K/D/R/P từ camera_info_manager.
     *     Nếu chưa load được thì tạo CameraInfo tối thiểu.
     *     Header luôn đồng bộ với Image.
     *
     * Output:
     *     CameraInfo message.
     */
    sensor_msgs::msg::CameraInfo createCameraInfoMessage(
        const std_msgs::msg::Header &imageHeader) const;

private:
    int width_;
    int height_;
    int fps_;
    bool useMjpeg_;

    std::string devicePath_;
    std::string frameId_;
    std::string topicName_;
    std::string cameraInfoTopic_;

    std::string cameraName_;
    std::string cameraInfoUrl_;

    GstElement *pipeline_;
    GstElement *appSink_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imagePub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfoPub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::unique_ptr<camera_info_manager::CameraInfoManager> cameraInfoManager_;
};

}  // namespace camera_usb_gst_cpp

#endif  // CAMERA_USB_GST_NODE_HPP_