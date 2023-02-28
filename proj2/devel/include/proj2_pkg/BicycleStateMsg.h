// Generated by gencpp from file proj2_pkg/BicycleStateMsg.msg
// DO NOT EDIT!


#ifndef PROJ2_PKG_MESSAGE_BICYCLESTATEMSG_H
#define PROJ2_PKG_MESSAGE_BICYCLESTATEMSG_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace proj2_pkg
{
template <class ContainerAllocator>
struct BicycleStateMsg_
{
  typedef BicycleStateMsg_<ContainerAllocator> Type;

  BicycleStateMsg_()
    : x(0.0)
    , y(0.0)
    , theta(0.0)
    , phi(0.0)  {
    }
  BicycleStateMsg_(const ContainerAllocator& _alloc)
    : x(0.0)
    , y(0.0)
    , theta(0.0)
    , phi(0.0)  {
  (void)_alloc;
    }



   typedef double _x_type;
  _x_type x;

   typedef double _y_type;
  _y_type y;

   typedef double _theta_type;
  _theta_type theta;

   typedef double _phi_type;
  _phi_type phi;





  typedef boost::shared_ptr< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> const> ConstPtr;

}; // struct BicycleStateMsg_

typedef ::proj2_pkg::BicycleStateMsg_<std::allocator<void> > BicycleStateMsg;

typedef boost::shared_ptr< ::proj2_pkg::BicycleStateMsg > BicycleStateMsgPtr;
typedef boost::shared_ptr< ::proj2_pkg::BicycleStateMsg const> BicycleStateMsgConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::proj2_pkg::BicycleStateMsg_<ContainerAllocator1> & lhs, const ::proj2_pkg::BicycleStateMsg_<ContainerAllocator2> & rhs)
{
  return lhs.x == rhs.x &&
    lhs.y == rhs.y &&
    lhs.theta == rhs.theta &&
    lhs.phi == rhs.phi;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::proj2_pkg::BicycleStateMsg_<ContainerAllocator1> & lhs, const ::proj2_pkg::BicycleStateMsg_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace proj2_pkg

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> >
{
  static const char* value()
  {
    return "7aab3456a3691108bb0b44c570431f4c";
  }

  static const char* value(const ::proj2_pkg::BicycleStateMsg_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x7aab3456a3691108ULL;
  static const uint64_t static_value2 = 0xbb0b44c570431f4cULL;
};

template<class ContainerAllocator>
struct DataType< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> >
{
  static const char* value()
  {
    return "proj2_pkg/BicycleStateMsg";
  }

  static const char* value(const ::proj2_pkg::BicycleStateMsg_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# The state of a bicycle model robot (x, y, theta, phi)\n"
"float64 x\n"
"float64 y\n"
"float64 theta\n"
"float64 phi\n"
;
  }

  static const char* value(const ::proj2_pkg::BicycleStateMsg_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.x);
      stream.next(m.y);
      stream.next(m.theta);
      stream.next(m.phi);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct BicycleStateMsg_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::proj2_pkg::BicycleStateMsg_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::proj2_pkg::BicycleStateMsg_<ContainerAllocator>& v)
  {
    s << indent << "x: ";
    Printer<double>::stream(s, indent + "  ", v.x);
    s << indent << "y: ";
    Printer<double>::stream(s, indent + "  ", v.y);
    s << indent << "theta: ";
    Printer<double>::stream(s, indent + "  ", v.theta);
    s << indent << "phi: ";
    Printer<double>::stream(s, indent + "  ", v.phi);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PROJ2_PKG_MESSAGE_BICYCLESTATEMSG_H
