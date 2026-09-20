"""The ROS half: a node beside the graph, a socket toward the pipeline.

Runs on the SYSTEM python3.10 (`/opt/ros/humble` is built against it), never
inside a conda env -- `scripts/ros2/bridge.sh` is the only supported way to
start it. Everything it does is mechanical: cache the newest synchronised
frame, hand goals to `NavigateToPose`, meter a discrete move on cmd_vel, and
pass the operator's floor switch through. No decisions, no conversions; the
pipeline side owns both (osg/ros2/frames.py).

Images are decoded with numpy rather than cv_bridge, and quaternions turned by
hand rather than by tf_transformations, so the image needs only
`ros-humble-ros-base` plus the message packages.
"""
from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import numpy as np

from . import frames as F
from .wire import serve_once

# ROS encodings this bridge can decode. Anything else is refused loudly: a
# silently mis-decoded image is a map of a room that does not exist.
_RGB_ENCODINGS = {"rgb8": (3, False), "bgr8": (3, True), "rgba8": (4, False),
                  "bgra8": (4, True), "mono8": (1, False)}
_DEPTH_ENCODINGS = {"16UC1": np.uint16, "mono16": np.uint16, "32FC1": np.float32}


def image_to_array(msg) -> np.ndarray:
    """`sensor_msgs/Image` -> ndarray, without cv_bridge."""
    enc = str(msg.encoding)
    if enc in _DEPTH_ENCODINGS:
        dtype = _DEPTH_ENCODINGS[enc]
        a = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, msg.width)
        return a.copy()
    if enc not in _RGB_ENCODINGS:
        raise ValueError(f"unsupported image encoding {enc!r}")
    channels, bgr = _RGB_ENCODINGS[enc]
    a = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    a = a.reshape(msg.height, msg.width, channels)
    if channels == 1:
        a = np.repeat(a, 3, axis=2)
    else:
        a = a[..., :3]
        if bgr:
            a = a[..., ::-1]
    return np.ascontiguousarray(a)


def stamp_to_float(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class BridgeNode:
    def __init__(self, args) -> None:
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from geometry_msgs.msg import Twist
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import Int32
        from tf2_ros import Buffer, TransformListener

        import message_filters

        self.args = args
        self.node = Node("osg_bridge")
        self._lock = threading.Condition()
        self._frame: Optional[dict] = None
        self._seq = 0
        self._info = None
        self._floor: Optional[int] = None
        self._goal_handle = None
        self._goal_id = 0
        self._goal_state = "idle"
        self._distance = float("nan")
        self.last_goal: Optional[dict] = None
        # Where the head is pointing. look_up/look_down are relative, so the
        # bridge has to remember; the Stretch reports it on /joint_states too,
        # but only once the controller is up.
        self._tilt = 0.0

        self.tf_buffer = Buffer()
        self._tf_listener = TransformListener(self.tf_buffer, self.node)

        qos = 10
        rgb = message_filters.Subscriber(self.node, Image, args.rgb_topic)
        depth = message_filters.Subscriber(self.node, Image, args.depth_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, depth], queue_size=qos, slop=float(args.sync_slop_s))
        self._sync.registerCallback(self._on_images)
        self.node.create_subscription(
            CameraInfo, args.camera_info_topic, self._on_info, qos)
        self.node.create_subscription(Int32, args.floor_topic, self._on_floor, qos)

        self._cmd_vel = self.node.create_publisher(Twist, args.cmd_vel_topic, qos)
        self._nav = ActionClient(self.node, NavigateToPose, args.nav_action)
        self._head = ActionClient(
            self.node, FollowJointTrajectory, args.head_traj_action)
        self._Twist = Twist
        self._NavigateToPose = NavigateToPose
        self._FollowJointTrajectory = FollowJointTrajectory
        self._rclpy = rclpy

    # ------------------------------------------------------------ subscribers

    def _on_info(self, msg) -> None:
        self._info = (np.asarray(msg.k, dtype=float).copy(), int(msg.width), int(msg.height))

    def _on_floor(self, msg) -> None:
        with self._lock:
            self._floor = int(msg.data)
            self._lock.notify_all()
        self.node.get_logger().info(f"floor switch -> {int(msg.data)}")

    def _on_images(self, rgb_msg, depth_msg) -> None:
        if self._info is None:
            return  # CameraInfo is latched-ish; a frame without it has no K
        frame_id = str(self.args.camera_frame or rgb_msg.header.frame_id)
        stamp = stamp_to_float(rgb_msg.header.stamp)
        try:
            T_map_cam = self._lookup(frame_id, rgb_msg.header.stamp)
        except Exception as exc:  # noqa: BLE001 -- a missing TF is normal at startup
            self.node.get_logger().warn(f"no {self.args.map_frame} -> {frame_id}: {exc}",
                                        throttle_duration_sec=5.0)
            return
        K, width, height = self._info
        with self._lock:
            self._seq += 1
            self._frame = {
                "seq": self._seq, "stamp": stamp,
                "rgb": image_to_array(rgb_msg),
                "depth": image_to_array(depth_msg),
                "depth_encoding": str(depth_msg.encoding),
                "K": K, "width": width, "height": height,
                "T_map_cam": T_map_cam, "frame_id": frame_id,
            }
            self._lock.notify_all()

    def _lookup(self, frame_id: str, stamp) -> np.ndarray:
        from rclpy.duration import Duration
        from rclpy.time import Time

        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.map_frame, frame_id, stamp,
                timeout=Duration(seconds=0.05))
        except Exception:
            # The exact stamp is not in the buffer yet. The latest is a better
            # answer than no frame at all -- at 30 Hz it is one frame stale.
            tf = self.tf_buffer.lookup_transform(
                self.args.map_frame, frame_id, Time())
        t, r = tf.transform.translation, tf.transform.rotation
        return F.transform_to_matrix((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))

    # -------------------------------------------------------------------- ops

    def ping(self) -> dict:
        with self._lock:
            seq = self._seq
        return {"node": "osg_bridge", "frames": seq,
                "ros_time": self.node.get_clock().now().nanoseconds * 1e-9,
                "has_camera_info": self._info is not None}

    def get_frame(self, min_stamp: float = 0.0, timeout_s: float = 5.0):
        deadline = time.time() + float(timeout_s)
        with self._lock:
            while True:
                frame = self._frame
                if frame is not None and frame["stamp"] >= float(min_stamp):
                    return frame
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    return None
                self._lock.wait(remaining)

    def send_goal(self, x: float, y: float, yaw: float, frame_id: str) -> dict:
        from geometry_msgs.msg import PoseStamped

        if not self._nav.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(
                f"no {self.args.nav_action} action server. Is Nav2 running?")
        pose = PoseStamped()
        pose.header.frame_id = str(frame_id)
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        qx, qy, qz, qw = F.yaw_to_quat(float(yaw))
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        goal = self._NavigateToPose.Goal()
        goal.pose = pose

        self._cancel_goal()
        self._goal_id += 1
        self._goal_state = "active"
        self.last_goal = {"x": float(x), "y": float(y), "yaw": float(yaw),
                          "frame_id": str(frame_id), "goal_id": self._goal_id}
        future = self._nav.send_goal_async(goal, feedback_callback=self._on_feedback)
        future.add_done_callback(self._on_goal_response)
        return {"goal_id": self._goal_id}

    def _on_feedback(self, msg) -> None:
        self._distance = float(getattr(msg.feedback, "distance_remaining", float("nan")))

    def _on_goal_response(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self._goal_state = "rejected"
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        from action_msgs.msg import GoalStatus

        status = future.result().status
        self._goal_state = {
            GoalStatus.STATUS_SUCCEEDED: "succeeded",
            GoalStatus.STATUS_ABORTED: "aborted",
            GoalStatus.STATUS_CANCELED: "canceled",
        }.get(status, "aborted")
        self._goal_handle = None

    def nav_status(self) -> dict:
        return {"state": self._goal_state, "goal_id": self._goal_id,
                "distance_remaining": self._distance}

    def cancel(self) -> dict:
        cancelled = self._cancel_goal()
        return {"cancelled": cancelled}

    def _cancel_goal(self) -> bool:
        handle, self._goal_handle = self._goal_handle, None
        if handle is not None:
            handle.cancel_goal_async()
        if self._goal_state == "active":
            self._goal_state = "canceled"
        self._stop_base()
        return handle is not None

    # ------------------------------------------------- the FSM's own actions

    def execute(self, action: str, forward_m: float, turn_deg: float) -> dict:
        """One discrete move, metered against odometry.

        Closed on TF rather than timed, because a timed open-loop move drifts
        with battery and carpet, and the pipeline's `forward_m` is a distance
        the costmap and the frontier reach radius both assume.
        """
        if action == "move_forward":
            return self._drive(distance_m=float(forward_m), yaw_deg=0.0)
        if action == "turn_left":
            return self._drive(distance_m=0.0, yaw_deg=float(turn_deg))
        if action == "turn_right":
            return self._drive(distance_m=0.0, yaw_deg=-float(turn_deg))
        raise ValueError(f"cannot execute {action!r} on the base")

    def _base_pose(self):
        from rclpy.time import Time

        tf = self.tf_buffer.lookup_transform(
            self.args.odom_frame, self.args.base_frame, Time())
        t, r = tf.transform.translation, tf.transform.rotation
        R = F.quat_to_matrix(r.x, r.y, r.z, r.w)
        return np.array([t.x, t.y]), float(np.arctan2(R[1, 0], R[0, 0]))

    def _drive(self, distance_m: float, yaw_deg: float) -> dict:
        start_xy, start_yaw = self._base_pose()
        target_yaw = np.radians(yaw_deg)
        twist = self._Twist()
        deadline = time.time() + float(self.args.move_timeout_s)
        achieved = 0.0
        rate = 0.05
        while time.time() < deadline:
            xy, yaw = self._base_pose()
            if distance_m > 0.0:
                achieved = float(np.linalg.norm(xy - start_xy))
                if achieved >= distance_m:
                    break
                twist.linear.x = float(self.args.linear_speed)
                twist.angular.z = 0.0
            else:
                achieved = float(np.arctan2(np.sin(yaw - start_yaw),
                                            np.cos(yaw - start_yaw)))
                if abs(achieved) >= abs(target_yaw) - 1e-3:
                    break
                twist.linear.x = 0.0
                twist.angular.z = float(np.sign(target_yaw) * self.args.angular_speed)
            self._cmd_vel.publish(twist)
            time.sleep(rate)
        self._stop_base()
        return {"achieved": achieved, "timed_out": time.time() >= deadline}

    def _stop_base(self) -> None:
        self._cmd_vel.publish(self._Twist())

    def look(self, tilt_delta_deg: float) -> dict:
        """Tilt the head. `stair_sense` and the close look both use this."""
        from builtin_interfaces.msg import Duration as MsgDuration
        from trajectory_msgs.msg import JointTrajectoryPoint

        if not self._head.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"no {self.args.head_traj_action} action server")
        target = self._tilt + np.radians(float(tilt_delta_deg))
        goal = self._FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [self.args.head_tilt_joint]
        point = JointTrajectoryPoint()
        point.positions = [float(target)]
        point.time_from_start = MsgDuration(sec=1)
        goal.trajectory.points = [point]
        self._head.send_goal_async(goal)
        self._tilt = float(target)
        return {"tilt_deg": float(np.degrees(target))}

    def pop_floor_switch(self) -> dict:
        with self._lock:
            floor, self._floor = self._floor, None
        return {"floor": floor}

    def get_last_goal(self):
        return self.last_goal

    def handlers(self) -> dict:
        return {
            "ping": self.ping, "get_frame": self.get_frame,
            "send_goal": self.send_goal, "nav_status": self.nav_status,
            "cancel": self.cancel, "execute": self.execute, "look": self.look,
            "pop_floor_switch": self.pop_floor_switch, "last_goal": self.get_last_goal,
        }


def serve(bridge, addr, authkey) -> None:
    """Accept pipeline connections until the process is killed.

    One client at a time: the pipeline is a single control loop, and a second
    connection would mean two things steering one robot.
    """
    from multiprocessing.connection import Listener

    with Listener(addr, authkey=authkey) as listener:
        bridge.node.get_logger().info(f"bridge listening on {addr}")
        while True:
            conn = listener.accept()
            bridge.node.get_logger().info("pipeline connected")
            try:
                while serve_once(conn, bridge.handlers()):
                    pass
            finally:
                conn.close()
                bridge.node.get_logger().info("pipeline disconnected")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge-addr", default="127.0.0.1:18765")
    p.add_argument("--authkey", default="osg-ros2")
    p.add_argument("--rgb-topic", default="/camera/color/image_raw")
    p.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    p.add_argument("--sync-slop-s", type=float, default=0.05)
    p.add_argument("--map-frame", default="map")
    p.add_argument("--camera-frame", default="")
    p.add_argument("--nav-action", default="navigate_to_pose")
    p.add_argument("--cmd-vel-topic", default="/stretch/cmd_vel")
    p.add_argument("--odom-frame", default="odom")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--linear-speed", type=float, default=0.15)
    p.add_argument("--angular-speed", type=float, default=0.5)
    p.add_argument("--move-timeout-s", type=float, default=10.0)
    p.add_argument("--head-traj-action",
                   default="/stretch_controller/follow_joint_trajectory")
    p.add_argument("--head-tilt-joint", default="joint_head_tilt")
    p.add_argument("--floor-topic", default="/osg/floor")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from .wire import parse_addr

    args = parse_args(argv)
    rclpy.init()
    bridge = BridgeNode(args)
    executor = MultiThreadedExecutor()
    executor.add_node(bridge.node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        serve(bridge, parse_addr(args.bridge_addr), args.authkey.encode("utf-8"))
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        bridge.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
