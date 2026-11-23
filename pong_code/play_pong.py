''' Pong Project
   Node:        /trajectory
   Publish:     /joint_states           sensor_msgs/JointState
                /pose                   geometry_msgs.msg/PoseStamped
                /twist                  geometry_msgs.msg/TwistStamped

'''

import rclpy
import numpy as np
import tf2_ros

from math               import pi, sin, cos, acos, atan2, sqrt, fmod, exp

from asyncio            import Future
from rclpy.node         import Node
from geometry_msgs.msg  import PoseStamped, TwistStamped
from geometry_msgs.msg  import TransformStamped
from sensor_msgs.msg    import JointState
from std_msgs.msg       import Header, Float64

# Grab the Utilities
from utils.TransformHelpers     import *
from utils.TrajectoryUtils      import *

# Grab the general fkin from HW5 P6.
from hw5code.KinematicChain     import KinematicChain

# ball stuff
from std_msgs.msg           import ColorRGBA
from visualization_msgs.msg import Marker



#
#   Trajectory Generator Node Class
#
#   This inherits all the standard ROS node stuff, but adds an
#   update() method to be called regularly by an internal timer and a
#   shutdown method to stop the timer.
#
#   Arguments are the node name and a future object (to force a shutdown).
#
class TrajectoryNode(Node):
    # Initialization.
    def __init__(self, name, future):
        # Initialize the node and store the future object (to end).
        super().__init__(name)
        self.future = future

        ##############################################################
        # INITIALIZE YOUR TRAJECTORY DATA!

        # Define the list of joint names MATCHING THE JOINT NAMES IN THE URDF!
        self.jointnames=['theta1','theta2','theta3','theta4','theta5','theta6']

        # Set up the kinematic chain object.
        self.chain = KinematicChain(self, 'world', 'tip', self.jointnames)

        # Define the matching initial joint/task positions.
        self.q0 = np.radians(np.array([0, 180, -90, 180, 0, 90]))
        (self.p0, self.R0, _, _) = self.chain.fkin(self.q0)

        # Use Newton Raphson to find joints at position of ball
        q = self.q0.copy()
        c = 0.1
        J = None

        # Final configuartions of paddle
        pball = np.array([0.50, 0.50, 0.50])
        Rd = np.eye(3)
        vF = np.array([0.0, 1.0, 0.25])
        wF = np.array([0.0, 0.0, 0.0])

        for i in range(100):
            p, R, Jv, Jw = self.chain.fkin(q)
            p_err = pball - p
            r_err = eR(Rd, R)
            e = np.concatenate((p_err, r_err))

            if np.linalg.norm(e) < 1e-3 and i > 0:
                break

            J = np.vstack((Jv, Jw))
            q += np.linalg.pinv(J) @ (c * e)


        self.qF = q.copy()
        self.qF_dot = np.linalg.pinv(J) @ np.concatenate((vF, wF))
        # stopping position in one second
        self.qEnd = self.qF + 0.5 * self.qF_dot
        
        self.qc = self.q0

        ### 
        # ball stuff
        ###

        # ball position + velo
        self.pball = pball.copy()
        self.vball = np.zeros(3)

        self.hit = False

        # ball and paddle dims
        self.rball = 0.02
        self.rpaddle = 0.08
        self.hpaddle = 0.016
        # bounce coeff
        self.e = 0.8
        self.grav = np.array([0.0, 0.0, -9.81])


        ##############################################################
        # Setup the logistics of the node:
        # Add publishers to send the joint and task commands.  Also
        # add a TF broadcaster, so the desired pose appears in RVIZ.
        self.pubjoint = self.create_publisher(JointState, '/joint_states', 10)
        self.pubpose  = self.create_publisher(PoseStamped, '/pose', 10)
        self.pubtwist = self.create_publisher(TwistStamped, '/twist', 10)
        self.tfbroad  = tf2_ros.TransformBroadcaster(self)

        # ball publish
        self.pubball = self.create_publisher(Marker, '/visualization_marker', 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while(not self.count_subscribers('/joint_states')):
            pass

        # Set up the timer to update at 100Hz, with (t=0) occuring in
        # the first update cycle (dt) from now.
        self.dt    = 0.01                       # 100Hz.
        self.t     = -self.dt                   # Seconds since start
        self.now   = self.get_clock().now()     # ROS time since 1970
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, 1/self.dt))

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()


    # Update - send a new joint command every time step.
    def update(self):
        # Increment time.  We do so explicitly to avoid system jitter.
        self.t   = self.t   + self.dt
        self.now = self.now + rclpy.time.Duration(seconds=self.dt)

        ##############################################################
        # COMPUTE THE TRAJECTORY AT THIS TIME INSTANCE.

        # # Stop after 15 sec.
        if self.t > 6.0:
            self.future.set_result("Trajectory has ended")
            return
        
        # Hitting the ball, then slowing the paddle down to zero, then returning to home
        if self.t < 3.0:
            qc, qcdot = spline(self.t, 3.0, self.q0, self.qF, np.zeros(6), self.qF_dot)

            pd, Rd, Jv, Jw = self.chain.fkin(qc)

            vd = Jv @ qcdot
            wd = Jw @ qcdot

        elif self.t < 4.0:
            qc, qcdot = spline(self.t - 3.0, 1.0, self.qF, self.qEnd, self.qF_dot, np.zeros(6))

            pd, Rd, Jv, Jw = self.chain.fkin(qc)

            vd = Jv @ qcdot
            wd = Jw @ qcdot

        else:
            qc, qcdot = goto(self.t - 4.0, 2.0, self.qEnd, self.q0)

            pd, Rd, Jv, Jw = self.chain.fkin(qc)
            
            vd = Jv @ qcdot
            wd = Jw @ qcdot

    
        # Save the joint command position.
        self.qc = qc

        n = Rd[:,1] # vector normal to the tip (paddle)
        r = self.pball - pd # ball relative to paddle center
        dn = r @ n # how far the ball is from the paddle face - measured straight out
        dr = np.linalg.norm(r - dn * n) # how far from the middle of the paddle face the ball is - mesaured along the paddle
        vpad = vd + np.cross(wd, r) # paddle velo at the ball
        vn = (self.vball - vpad) @ n # velo of the ball relative to the paddle

        # if (ball touches the plane of the paddle) and (ball is within the area of the paddle face) and (ball and paddle are moving towards each other)
        if (abs(dn) < self.rball + self.hpaddle/2) and (dr < self.rpaddle) and (vn * dn < 0):
            self.vball = self.vball - (1 + self.e) * vn * n
            self.hit   = True
        # if already hit
        if self.hit:
            self.vball = self.vball + self.dt * self.grav
            self.pball = self.pball + self.dt * self.vball
            # hit the ground, so bounce
            if self.pball[2] < self.rball and self.vball[2] < 0:
                self.vball[2] = -self.e * self.vball[2]


        ##############################################################
        # Finish by publishing the data (joint and task commands).
        #  qc and qcdot = Joint Commands  as  /joint_states  to view/plot
        #  pd and Rd    = Task pos/orient as  /pose & TF     to view/plot
        #  vd and wd    = Task velocities as  /twist         to      plot
        header=Header(stamp=self.now.to_msg(), frame_id='world')
        self.pubjoint.publish(JointState(
            header=header,
            name=self.jointnames,
            position=qc.tolist(),
            velocity=qcdot.tolist()))
        self.pubpose.publish(PoseStamped(
            header=header,
            pose=Pose_from_Rp(Rd,pd)))
        self.pubtwist.publish(TwistStamped(
            header=header,
            twist=Twist_from_vw(vd,wd)))
        self.tfbroad.sendTransform(TransformStamped(
            header=header,
            child_frame_id='desired',
            transform=Transform_from_Rp(Rd,pd)))


        # publish the ball to rviz
        marker = Marker()
        marker.header = header
        marker.ns = 'ball'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point_from_p(self.pball)
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=2*self.rball, y=2*self.rball, z=2*self.rball)
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        self.pubball.publish(marker)


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Create a future object to signal when the trajectory ends.
    future = Future()

    # Initialize the trajectory generator node.
    trajectory = TrajectoryNode('trajectory', future)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory is
    # complete (as signaled by the future object).
    rclpy.spin_until_future_complete(trajectory, future)

    # Report the reason for shutting down.
    if future.done():
        trajectory.get_logger().info("Stopping: " + future.result())
    else:
        trajectory.get_logger().info("Stopping: Interrupted")

    # Shutdown the node and ROS.
    trajectory.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
