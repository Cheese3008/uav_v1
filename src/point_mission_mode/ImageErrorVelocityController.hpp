#pragma once

#include "ControlTypes.hpp"

namespace point_mission_mode
{
class ImageErrorVelocityController
{
public:
    void configure(const ImageErrorControllerParams &params);
    void reset();
    ImageErrorControllerOutput update(const ImageErrorControllerInput &input) const;

private:
    ImageErrorControllerParams params_{};
};
} // namespace point_mission_mode
