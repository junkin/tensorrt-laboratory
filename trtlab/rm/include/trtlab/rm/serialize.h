#pragma once

struct ISerializableResource
{
    virtual bool        serializable()         = 0;
    virtual bool        is_serialized()        = 0;
    virtual void        serialize()   = 0;
    virtual void        deserialize() = 0;
    virtual std::size_t serialize_size()       = 0;
};

// private impl

struct SerializableResource
{
    bool serializable() final override { return true; }
};

struct UnserializableResource final
{
    bool serializable() final override { return false; }
    bool is_serialized() final override { return false; }
    void serialize() final override { }
    void deserialize() final override { }
    std::size_t serialized_size() final override { return 0; }
};