// Generated by gencpp from file stdr_msgs/LoadMap.msg
// DO NOT EDIT!


#ifndef STDR_MSGS_MESSAGE_LOADMAP_H
#define STDR_MSGS_MESSAGE_LOADMAP_H

#include <ros/service_traits.h>


#include <stdr_msgs/LoadMapRequest.h>
#include <stdr_msgs/LoadMapResponse.h>


namespace stdr_msgs
{

struct LoadMap
{

typedef LoadMapRequest Request;
typedef LoadMapResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct LoadMap
} // namespace stdr_msgs


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::stdr_msgs::LoadMap > {
  static const char* value()
  {
    return "656f50b5e78328d62ac1c4a5c19fefd5";
  }

  static const char* value(const ::stdr_msgs::LoadMap&) { return value(); }
};

template<>
struct DataType< ::stdr_msgs::LoadMap > {
  static const char* value()
  {
    return "stdr_msgs/LoadMap";
  }

  static const char* value(const ::stdr_msgs::LoadMap&) { return value(); }
};


// service_traits::MD5Sum< ::stdr_msgs::LoadMapRequest> should match
// service_traits::MD5Sum< ::stdr_msgs::LoadMap >
template<>
struct MD5Sum< ::stdr_msgs::LoadMapRequest>
{
  static const char* value()
  {
    return MD5Sum< ::stdr_msgs::LoadMap >::value();
  }
  static const char* value(const ::stdr_msgs::LoadMapRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::stdr_msgs::LoadMapRequest> should match
// service_traits::DataType< ::stdr_msgs::LoadMap >
template<>
struct DataType< ::stdr_msgs::LoadMapRequest>
{
  static const char* value()
  {
    return DataType< ::stdr_msgs::LoadMap >::value();
  }
  static const char* value(const ::stdr_msgs::LoadMapRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::stdr_msgs::LoadMapResponse> should match
// service_traits::MD5Sum< ::stdr_msgs::LoadMap >
template<>
struct MD5Sum< ::stdr_msgs::LoadMapResponse>
{
  static const char* value()
  {
    return MD5Sum< ::stdr_msgs::LoadMap >::value();
  }
  static const char* value(const ::stdr_msgs::LoadMapResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::stdr_msgs::LoadMapResponse> should match
// service_traits::DataType< ::stdr_msgs::LoadMap >
template<>
struct DataType< ::stdr_msgs::LoadMapResponse>
{
  static const char* value()
  {
    return DataType< ::stdr_msgs::LoadMap >::value();
  }
  static const char* value(const ::stdr_msgs::LoadMapResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // STDR_MSGS_MESSAGE_LOADMAP_H
