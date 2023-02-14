#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

import rospy
import tf
import tf2_ros

"""
Set of classes for defining SE(3) trajectories for the end effector of a robot 
manipulator
"""

class Trajectory:

    def __init__(self, total_time):
        """
        Parameters
        ----------
        total_time : float
            desired duration of the trajectory in seconds 
        """
        self.total_time = total_time

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.
        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 
        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        pass

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.
        The function get_g_matrix from utils may be useful to perform some frame
        transformations.
        Parameters
        ----------
        time : float
        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        pass

    def display_trajectory(self, num_waypoints=67, show_animation=False, save_animation=False):
        """
        Displays the evolution of the trajectory's position and body velocity.
        Parameters
        ----------
        num_waypoints : int
            number of waypoints in the trajectory
        show_animation : bool
            if True, displays the animated trajectory
        save_animatioon : bool
            if True, saves a gif of the animated trajectory
        """
        trajectory_name = self.__class__.__name__
        times = np.linspace(0, self.total_time, num=num_waypoints)
        target_positions = np.vstack([self.target_pose(t)[:3] for t in times])
        target_velocities = np.vstack([self.target_velocity(t)[:3] for t in times])
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        colormap = plt.cm.brg(np.fmod(np.linspace(0, 1, num=num_waypoints), 1))

        # Position plot
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        pos_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax0.set_xlim3d([min(target_positions[:, 0]) + pos_padding[0][0], 
                        max(target_positions[:, 0]) + pos_padding[0][1]])
        ax0.set_xlabel('X')
        ax0.set_ylim3d([min(target_positions[:, 1]) + pos_padding[1][0], 
                        max(target_positions[:, 1]) + pos_padding[1][1]])
        ax0.set_ylabel('Y')
        ax0.set_zlim3d([min(target_positions[:, 2]) + pos_padding[2][0], 
                        max(target_positions[:, 2]) + pos_padding[2][1]])
        ax0.set_zlabel('Z')
        ax0.set_title("%s evolution of\nend-effector's position." % trajectory_name)
        line0 = ax0.scatter(target_positions[:, 0], 
                        target_positions[:, 1], 
                        target_positions[:, 2], 
                        c=colormap,
                        s=2)

        # Velocity plot
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        vel_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax1.set_xlim3d([min(target_velocities[:, 0]) + vel_padding[0][0], 
                        max(target_velocities[:, 0]) + vel_padding[0][1]])
        ax1.set_xlabel('X')
        ax1.set_ylim3d([min(target_velocities[:, 1]) + vel_padding[1][0], 
                        max(target_velocities[:, 1]) + vel_padding[1][1]])
        ax1.set_ylabel('Y')
        ax1.set_zlim3d([min(target_velocities[:, 2]) + vel_padding[2][0], 
                        max(target_velocities[:, 2]) + vel_padding[2][1]])
        ax1.set_zlabel('Z')
        ax1.set_title("%s evolution of\nend-effector's translational body-frame velocity." % trajectory_name)
        line1 = ax1.scatter(target_velocities[:, 0], 
                        target_velocities[:, 1], 
                        target_velocities[:, 2], 
                        c=colormap,
                        s=2)

        if show_animation or save_animation:
            def func(num, line):
                line[0]._offsets3d = target_positions[:num].T
                line[0]._facecolors = colormap[:num]
                line[1]._offsets3d = target_velocities[:num].T
                line[1]._facecolors = colormap[:num]
                return line

            # Creating the Animation object
            line_ani = animation.FuncAnimation(fig, func, frames=num_waypoints, 
                                                          fargs=([line0, line1],), 
                                                          interval=max(1, int(1000 * self.total_time / (num_waypoints - 1))), 
                                                          blit=False)
        plt.show()
        if save_animation:
            line_ani.save('%s.gif' % trajectory_name, writer='pillow', fps=60)
            print("Saved animation to %s.gif" % trajectory_name)

class LinearTrajectory(Trajectory):

    def __init__(self, start_point, goal_point, total_time):
        """
        Remember to call the constructor of Trajectory
        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """
        Trajectory.__init__(self, total_time)
        self.start_point = start_point
        self.goal_point = goal_point


    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.
        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 
        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        pose = np.hstack((self.goal_point, np.array([0, 1, 0, 0])))
        if time < self.total_time:
            vec = self.goal_point - self.start_point
            dist = np.linalg.norm(vec)
            unit_vec = (vec)/dist
            a = 6 * dist/(self.total_time**3)
            x = -a*((time**3)/3) + a*self.total_time*((time**2)/2)
            position = self.start_point + x*unit_vec
            position = np.reshape(position, (3,))
            pose = np.hstack((position, np.array([0, 1, 0, 0])))
        return pose

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.
        The function get_g_matrix from utils may be useful to perform some frame
        transformations.
        Parameters
        ----------
        time : float
        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        twist = np.zeros(6)
        if time < self.total_time:
            dist = np.linalg.norm(self.goal_point - self.start_point)
            a = 6 * dist/(self.total_time**3)
            vt = -a * time * (time - self.total_time)
            unit_vec = (self.goal_point - self.start_point)/dist
            v = vt*unit_vec
            v = v.reshape((3,))
            twist = np.hstack((v, np.zeros(3)))
        return twist
        

class CircularTrajectory(Trajectory):

    def __init__(self, start_point, center_position, radius, circle_time, reach_time):
        """
        Remember to call the constructor of Trajectory
        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """
        Trajectory.__init__(self, circle_time+reach_time)
        self.center_position = center_position
        self.radius = radius
        self.circle_time = circle_time
        self.reach_time = reach_time
        self.z = self.center_position[2]

        self.circle_start = np.array([self.center_position[0] + self.radius, self.center_position[1], self.z])
        self.reach_trajectory = LinearTrajectory(start_point, self.circle_start, reach_time)

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.
        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 
        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        pose = np.hstack((self.circle_start, np.array([0, 1, 0, 0])))
        if time < self.reach_time:
            pose = self.reach_trajectory.target_pose(time)
        elif time < self.total_time:
            time = time - self.reach_time

            total_sweep = 2*np.pi
            a = 6 * total_sweep/(self.circle_time**3)
            theta = -a*((time**3)/3 - self.circle_time*(time**2)/2)

            x = self.center_position[0] + self.radius * np.cos(theta)
            y = self.center_position[1] + self.radius * np.sin(theta)

            position = np.array([x, y, self.z])
            pose = np.hstack((position, np.array([0, 1, 0, 0])))
        return pose

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.
        The function get_g_matrix from utils may be useful to perform some frame
        transformations.
        Parameters
        ----------
        time : float
        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        twist = np.zeros(6)
        if time < self.reach_time:
            twist = self.reach_trajectory.target_velocity(time)
        elif time < self.total_time:
            time = time - self.reach_time

            total_sweep = 2*np.pi
            a = 6 * total_sweep/(self.circle_time**3)
            theta_dot = -a*time**2 + a*time*self.circle_time
            w = theta_dot*np.array([0, 0, 1])

            theta = -a*((time**3)/3 - self.circle_time*(time**2)/2)
            x_dot = -self.radius*theta_dot*np.sin(theta)
            y_dot = self.radius*theta_dot*np.cos(theta)
            twist = np.array([x_dot, y_dot, 0, 0, 0, 0])
        return twist

class PolygonalTrajectory(Trajectory):
    def __init__(self, points, total_time):
        """
        Remember to call the constructor of Trajectory.
        You may wish to reuse other trajectories previously defined in this file.
        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """
        Trajectory.__init__(self, total_time)
        num_points = len(points)
        self.time_segments = np.linspace(0, total_time, num_points)
        self.trajectories = [None] * (len(self.time_segments)-1)
        for i in range(1, len(self.time_segments)):
            segment_time = self.time_segments[i] - self.time_segments[i-1]
            start_point = points[i-1]
            goal_point = points[i]
            self.trajectories[i-1] = LinearTrajectory(start_point, goal_point, segment_time)

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.
        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 
        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        seg_idx = 0
        seg_t = time
        for i in range(len(self.time_segments)-1):
            if time <= self.time_segments[i+1]:
                seg_idx = i
                seg_t = time - self.time_segments[i]
                break
        pose = self.trajectories[seg_idx].target_pose(seg_t)
        return pose
        
    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.
        The function get_g_matrix from utils may be useful to perform some frame
        transformations.
        Parameters
        ----------
        time : float
        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        seg_idx = 0
        seg_t = time
        for i in range(len(self.time_segments)-1):
            if time <= self.time_segments[i+1]:
                seg_idx = i
                seg_t = time - self.time_segments[i]
                break
        twist = self.trajectories[seg_idx].target_velocity(seg_t)
        return twist

if __name__ == '__main__':
    """
    Run this file to visualize plots of your paths. Note: the provided function
    only visualizes the end effector position, not its orientation. Use the 
    animate function to visualize the full trajectory in a 3D plot.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle, polygon.  Default: line'
    )
    parser.add_argument('--animate', action='store_true', help=
        'If you set this flag, the animated trajectory will be shown.'
    )
    args = parser.parse_args()

    trajectory = define_trajectories(args)
    
    if trajectory:
        trajectory.display_trajectory(show_animation=args.animate)