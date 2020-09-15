
#pragma once

#include <cstdint>

struct IContext
{
    virtual void append_audio(const std::int16_t* audio, std::size_t sample_count) = 0;
    virtual void reset() = 0;
};
