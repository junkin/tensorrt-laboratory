#pragma once

struct ICachableResource
{
    virtual bool        cachable()      = 0;
    virtual bool        is_cached()     = 0;
    virtual void        reserve()       = 0;
    virtual void        shrink_to_fit() = 0;
    virtual std::size_t cache_size()    = 0;
};

// private impl

struct CachableResource
{
    bool cachable() final override
    {
        return true;
    }
};

struct UncachableResource final
{
    bool cachable() final override
    {
        return false;
    }
    bool is_cached() final override
    {
        return false;
    }
    void        reserve() final override {}
    void        shrink_to_fit() final override {}
    std::size_t cache_size() final override
    {
        return 0;
    }
};