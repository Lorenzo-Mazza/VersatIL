import rospy
from testbed_msgs.msg import RobotStateRcm, RcmPose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from PIL import Image as PILImage
import cv2
import sys
import signal
import torchvision.transforms as transforms
import torch
import pickle
import os
from scipy.spatial.transform import Rotation as R
import zmq
import io
import argparse
from legacy_config import TaskBaseConfig, DiffusionConfig
from workspace import DiffusionWorkspace
import time


def vis_disparity(
    disp,
    min_val=None,
    max_val=None,
    invalid_thres=np.inf,
    color_map=cv2.COLORMAP_TURBO,
    cmap=None,
    other_output={},
):
    """
    @disp: np array (H,W)
    @invalid_thres: > thres is invalid
    """
    disp = disp.copy()
    H, W = disp.shape[:2]
    invalid_mask = disp >= invalid_thres
    if (invalid_mask == 0).sum() == 0:
        other_output["min_val"] = None
        other_output["max_val"] = None
        return np.zeros((H, W, 3))
    if min_val is None:
        min_val = disp[invalid_mask == 0].min()
    if max_val is None:
        max_val = disp[invalid_mask == 0].max()
    other_output["min_val"] = min_val
    other_output["max_val"] = max_val
    vis = ((disp - min_val) / (max_val - min_val)).clip(0, 1) * 255
    if cmap is None:
        vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[..., ::-1]
    else:
        vis = cmap(vis.astype(np.uint8))[..., :3] * 255
    if invalid_mask.any():
        vis[invalid_mask] = 0
    return vis.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference client for robot control")
    parser.add_argument(
        "--server_address",
        type=str,
        default="tcp://localhost:5555",
        help="Address of the observation server",
    )
    parser.add_argument(
        "--update_interval",
        type=float,
        default=0.7,
        help="Time interval between updates in seconds",
    )
    return parser.parse_args()


def robot_to_camera_space(robot_space_pos, rotation) -> list[float]:
    return (
        np.array(robot_space_pos) + np.dot(rotation, np.array(robot_space_pos))
    ).tolist()


def camera_to_robot_space(camera_space_pos, rotation) -> list[float]:
    return (
        np.array(camera_space_pos) + np.dot(rotation.T, np.array(camera_space_pos))
    ).tolist()


class Model:
    def __init__(self, config: TaskBaseConfig):
        print(config)
        self.model = DiffusionWorkspace(config=config)
        # Our best model so far.
        self.model.load_checkpoint(
            path="/mnt/cluster/workspaces/mazzalore/iros/diffusion_policy_checkpoints/experiment_20250515_113430/latest.pt"
        )
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(540),
                transforms.Resize(config.image_size),
                transforms.ToTensor(),
            ]
        )
        self.max_timesteps = 10000
        self.num_queries = self.model.config.action_horizon
        self.action_dimension = self.model.config.action_dim
        self.all_time_actions = torch.zeros(
            [
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.action_dimension,
            ]
        ).cuda()
        self.timestep = 0

        self.predict_global_position = True
        self.predict_next_position = True
        if self.predict_next_position:
            self.predict_action = False
        else:
            self.predict_action = True

    def get_action(
        self,
        post: list[tuple[float, float, float]],
        left_img: list,
        right_img: list,
        depth_img: list,
    ):
        """Returns a (dx, dy, dz) action prediction."""

        left_img = [self.transform(img) for img in left_img]
        right_img = [self.transform(img) for img in right_img]
        depth_img = [self.transform(img) for img in depth_img]
        # Normalization already takes place inside the model, so we don't need to do it here.

        left_img = torch.stack(left_img).unsqueeze(0)  # Shape becomes (1, 2, 3, H, W)
        right_img = torch.stack(right_img).unsqueeze(0)  # Shape becomes (1, 2, 3, H, W)
        depth_img = torch.stack(depth_img).unsqueeze(0)  # Shape becomes (1, 2, 1, H, W)

        # Create the positional tensor and add the batch dimension.
        post_tensor = torch.tensor(post, dtype=torch.float32)
        post_tensor = post_tensor.unsqueeze(0)  # Shape becomes (1, 2, 6)

        obs_dict = {
            "image_source_1": left_img,  # should be (1, 2, 3, H, W)
            "image_source_2": right_img,  # should be (1,2, 3, H, W)
            "depth": depth_img,  # should be (1, 2, 1, H, W)
            "agent_pos": post_tensor,  # should be (1, 2, 3,)
        }
        print(right_img.shape)
        print(left_img.shape)
        print(post_tensor.shape)
        actions = self.model.predict_actions(obs_dict=obs_dict)
        # raw_action = actions.sum(dim=0).sum(dim=0).squeeze(0).cpu().detach().numpy()
        self.all_time_actions[
            [self.timestep], self.timestep : self.timestep + self.num_queries
        ] = actions
        actions_for_curr_step = self.all_time_actions[:, self.timestep]
        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        raw_action = raw_action.squeeze(0).cpu().detach().numpy()
        self.timestep += 1
        print(raw_action)
        return raw_action.tolist()


class RosSubscriber:
    def __init__(self):
        # Parse command line arguments
        args = parse_args()
        # TODO: remember to recalibrate this quaternion for every set of tests you run.
        camera_quaternion = [0.80830982, -0.05899544, 0.02594916, 0.5807681]
        self.rotation = R.from_quat(camera_quaternion)
        self.rotation = self.rotation.as_matrix()
        self.is_initialized = False

        # Robot callback storage
        self.robot_state = None
        self.initial_tip_pos_camera_space = None
        self.relative_tip_position_robot_space = None
        self.relative_tip_position_camera_space = None

        config = DiffusionConfig()
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(config=config)

        # Image callback storage
        self.left_img = None
        self.right_img = None

        # Data buffer to pass to the model
        self.left_img_buffer = []
        self.right_img_buffer = []
        self.depth_img_buffer = []
        self.tip_position_camera_space_buffer = []

        self.bridge = CvBridge()
        self.counter = 0

        # Initialize ZMQ connection to the observation server
        self.context = zmq.Context()
        self.socket = self.context.socket(
            zmq.REQ
        )  # REQ socket for request-reply pattern
        self.server_address = args.server_address
        print(f"Connecting to observation server at {self.server_address}")
        self.socket.connect(self.server_address)

        # Initialize ROS node and subscribers
        rospy.init_node("inference_client", anonymous=True)

        # Subscribe to robot state
        rospy.Subscriber(
            "/ur5e_rcm_controller/robotStateRcm",
            RobotStateRcm,
            self.robot_state_callback,
        )

        # Subscribe to camera images
        rospy.Subscriber(
            "/laparoscope/camera/left/image_raw", Image, self.left_img_callback
        )
        rospy.Subscriber(
            "/laparoscope/camera/right/image_raw", Image, self.right_img_callback
        )

        # Publisher for robot control
        self.rcm_publisher = rospy.Publisher(
            "/ur5e_rcm_controller/set_rcm_pose", RcmPose, queue_size=0
        )

        # Global position
        self.GLOBAL_ROBOT_POS = None

        # Debugging: save directory for images
        self.save_dir = "/mnt/cluster/temp/threading_il/current_observation"
        os.makedirs(self.save_dir, exist_ok=True)

        # Register shutdown handler
        signal.signal(signal.SIGINT, self.shutdown)

        while True:
            # Continuously update the robot position.
            self.update_loop()
            time.sleep(0.5)

        print("ROS subscribers initialized")
        rospy.spin()

    def robot_state_callback(self, msg):
        """Callback for robot state updates"""
        self.robot_state = msg
        self.relative_tip_position_robot_space = [
            msg.relativeTipPosition.x,
            msg.relativeTipPosition.y,
            msg.relativeTipPosition.z,
        ]
        self.relative_tip_position_camera_space = robot_to_camera_space(
            self.relative_tip_position_robot_space, self.rotation
        )

        if self.is_initialized is False:
            print("setting initial pos in camera space")
            self.is_initialized = True
            self.initial_tip_pos_camera_space = self.relative_tip_position_camera_space

        # Always start from 0.,0.,0. at the beginning of the test, when we are predicting global position.
        if self.model.predict_global_position:
            temp = np.array(self.relative_tip_position_camera_space) - np.array(
                self.initial_tip_pos_camera_space
            )
            self.relative_tip_position_camera_space = temp.tolist()

    def left_img_callback(self, msg):
        """Callback for left camera image updates"""
        self.left_img = self.convert_ros_image(msg)

    def right_img_callback(self, msg):
        """Callback for right camera image updates"""
        self.right_img = self.convert_ros_image(msg)

    def convert_ros_image(self, ros_img):
        """Convert ROS image message to PIL image"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding="rgb8")
            # Convert OpenCV image to PIL Image
            pil_image = PILImage.fromarray(cv_image)
            return pil_image
        except Exception as e:
            print(f"Error converting image: {e}")
            return None

    def update_loop(self):
        """Main loop for updating observations and controlling the robot"""
        try:
            # Check if we have all required data
            if (
                self.left_img is None
                or self.right_img is None
                or self.relative_tip_position_robot_space is None
            ):
                return

            # Debug: save directory for images
            save_dir = "/mnt/cluster/temp/threading_il/current_observation"
            os.makedirs(save_dir, exist_ok=True)
            rel_tip_pos_camera_space = self.relative_tip_position_camera_space
            # Send images to the observation server to get disparity map
            result = self.get_depth_from_server(self.left_img, self.right_img)

            if result and "depth_map" in result:
                depth_map = result["depth_map"]

                left_rectified = PILImage.fromarray(result["left_rectified"])
                right_rectified = PILImage.fromarray(result["right_rectified"])
                depth_img_pil = PILImage.fromarray(vis_disparity(depth_map))
                # Optionally crop and resize
                depth_img_pil = depth_img_pil.crop((210, 0, 750, 540)).resize(
                    (512, 512)
                )

                # Save the original images for debugging
                left_rectified.save(f"{self.save_dir}/{self.counter}_left.png")
                right_rectified.save(f"{self.save_dir}/{self.counter}_right.png")
                depth_img_pil.save(f"{self.save_dir}/{self.counter}_depth.png")

                self.counter += 1

                # Add to the buffer
                self.left_img_buffer.append(left_rectified)
                self.right_img_buffer.append(right_rectified)
                self.depth_img_buffer.append(
                    PILImage.fromarray(depth_map.astype(np.float32))
                )
                self.tip_position_camera_space_buffer.append(rel_tip_pos_camera_space)

            else:
                print("Failed to get valid depth map from server")

            # Remove the oldest image from the buffer
            if len(self.left_img_buffer) > 2:
                self.left_img_buffer = self.left_img_buffer[1:]
                self.right_img_buffer = self.right_img_buffer[1:]
                self.depth_img_buffer = self.depth_img_buffer[1:]
                self.tip_position_camera_space_buffer = (
                    self.tip_position_camera_space_buffer[1:]
                )

            if len(self.left_img_buffer) > 1:
                prediction = self.model.get_action(
                    post=self.tip_position_camera_space_buffer,
                    left_img=self.left_img_buffer,
                    depth_img=self.depth_img_buffer,
                    right_img=self.right_img_buffer,
                )

                if self.model.predict_next_position:
                    self.move_robot_with_next_position(prediction)
                else:
                    self.move_robot_with_action(prediction)

        except Exception as e:
            print(f"Error in update loop: {e}")

    def get_depth_from_server(self, left_img, right_img):
        try:
            left_buffer = io.BytesIO()
            right_buffer = io.BytesIO()
            left_img.save(left_buffer, format="PNG")
            right_img.save(right_buffer, format="PNG")
            left_binary = left_buffer.getvalue()
            right_binary = right_buffer.getvalue()
            request = {
                "left_image": left_binary,
                "right_image": right_binary,
            }
            print("Sending images to observation server...")
            start_time = time.time()
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
            process_time = time.time() - start_time
            if "error" in response:
                print(f"Error from server: {response['error']}")
                return None
            depth_map = np.frombuffer(
                response["depth_map"], dtype=np.dtype(response["depth_dtype"])
            ).reshape(response["depth_shape"])

            left_rectified = np.array(response["left_image"])
            right_rectified = np.array(response["right_image"])

            return {
                "depth_map": depth_map,
                "left_rectified": left_rectified,
                "right_rectified": right_rectified,
                "server_time": response.get("processing_time", 0),
                "total_time": process_time,
            }

        except Exception as e:
            print(f"Error getting depth from server: {e}")
            return None

    def publishData(self, position):
        """Publish robot position command"""
        MOVEMENT_TIME = 2
        msg = RcmPose()
        msg.targetPosition.x = position[0]
        msg.targetPosition.y = position[1]
        msg.targetPosition.z = position[2]
        msg.targetRoll = 0  # GLOBAL_ROLL
        msg.jerkLimit = 10
        msg.accelerationLimit = 5
        msg.epsilon = 0.00000001
        msg.movementTime = MOVEMENT_TIME
        print(f"Publishing position: {position}")
        self.rcm_publisher.publish(msg)

    def move_robot_with_next_position(self, next_position_camera_space):
        """Move the robot to the predicted position at t+1"""

        if (
            self.GLOBAL_ROBOT_POS is None
            and self.relative_tip_position_robot_space is not None
        ):
            self.GLOBAL_ROBOT_POS = self.relative_tip_position_robot_space.copy()

        if self.GLOBAL_ROBOT_POS is None:
            print("Cannot move robot: position unknown")
            return

        print(f"next predicted position in camera space {next_position_camera_space} ")
        print(
            f"initial position of the episode in camera space {self.initial_tip_pos_camera_space}"
        )

        if self.model.predict_global_position:
            # Re-add initial position, when we are predicting global position.
            next_position_camera_space = np.array(
                next_position_camera_space
            ) + np.array(self.initial_tip_pos_camera_space)

        next_position_robot_space = camera_to_robot_space(
            next_position_camera_space, self.rotation
        )

        movement = np.array(next_position_robot_space) - np.array(self.GLOBAL_ROBOT_POS)

        print(f"{movement=}")
        if (
            abs(movement[0]) > 0.05
            or abs(movement[1]) > 0.05
            or abs(movement[2]) > 0.05
        ):
            print("Invalid movement - exceeds limits")
            return

        self.GLOBAL_ROBOT_POS[0] = next_position_robot_space[0]
        self.GLOBAL_ROBOT_POS[1] = next_position_robot_space[1]
        self.GLOBAL_ROBOT_POS[2] = next_position_robot_space[2]
        # self.publishData(self.GLOBAL_ROBOT_POS)

    def move_robot_with_action(self, action_camera_space):
        """Move the robot by applying the generated action"""

        if (
            self.GLOBAL_ROBOT_POS is None
            and self.relative_tip_position_robot_space is not None
        ):
            self.GLOBAL_ROBOT_POS = self.relative_tip_position_robot_space.copy()

        if self.GLOBAL_ROBOT_POS is None:
            print("Cannot move robot: position unknown")
            return
        action_camera_space = np.array(action_camera_space)
        print(f"action in camera space {action_camera_space} ")
        print(f"initial position in camera space {self.initial_tip_pos_camera_space}")

        action_robot_space = camera_to_robot_space(action_camera_space, self.rotation)

        print(f"Action: {action_robot_space}")
        if (
            abs(action_robot_space[0]) > 0.05
            or abs(action_robot_space[1]) > 0.05
            or abs(action_robot_space[2]) > 0.05
        ):
            print("Invalid action - exceeds limits")
            return

        self.GLOBAL_ROBOT_POS[0] += action_robot_space[0]
        self.GLOBAL_ROBOT_POS[1] += action_robot_space[1]
        self.GLOBAL_ROBOT_POS[2] += action_robot_space[2]
        # self.publishData(self.GLOBAL_ROBOT_POS)

    def shutdown(self, signum, frame):
        """Clean shutdown handler"""
        print("Shutting down client...")
        # Close ZMQ connection
        self.socket.close()
        self.context.term()
        # Shutdown ROS
        rospy.signal_shutdown("Manual shutdown")
        sys.exit(0)


if __name__ == "__main__":
    subscriber = RosSubscriber()
