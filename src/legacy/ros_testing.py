import rospy
from std_msgs.msg import Float32MultiArray
from testbed_msgs.msg import RcmPose
import numpy as np
from PIL import Image as PILImage
import sys
import signal
import torchvision.transforms as transforms
import torch
from scipy.spatial.transform import Rotation as R

from legacy_config import TaskBaseConfig, DiffusionConfig
from workspace import DiffusionWorkspace
from testbed_msgs.srv import ObservationState


def unpack_observations(obs: list[Float32MultiArray]) -> tuple[list[tuple], list[tuple], list[PILImage], list[PILImage]]:
    relative_tip_positions, absolute_tip_positions, left_frames, depth_frames = [], [], [], []
    for data in obs:
        relative_tip_positions.append(tuple(data[:3]))
        absolute_tip_positions.append(tuple(data[3:6]))
        left_frame = list(data[6: 6 + 960 * 540 * 3])
        left_frame = np.array(left_frame).reshape((540, 960, 3)).astype(np.uint8)
        depth_frame = list(data[6 + 960 * 540 * 3:])
        depth_frame = np.array(depth_frame).reshape((512, 512)).astype(np.uint8)
        left_frame = PILImage.fromarray(left_frame).crop((210, 0, 750, 540)).resize((512, 512))
        depth_frame = PILImage.fromarray(depth_frame).convert("RGB")
        left_frames.append(left_frame)
        depth_frames.append(depth_frame)
    return relative_tip_positions, absolute_tip_positions, left_frames, depth_frames


class Model:

    def __init__(self, config: TaskBaseConfig):
        self.model = DiffusionWorkspace(config=config)
        # Our best model so far.
        self.model.load_checkpoint(path="/mnt/cluster/workspaces/mazzalore/iros/experiment_20250404_190135/best_model.pt")
        self.transform = transforms.Compose([
            transforms.CenterCrop(540),
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
        ])
        self.max_timesteps = 10000
        self.num_queries = self.model.config.action_horizon
        self.action_dimension = self.model.config.action_dim
        self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.action_dimension]).cuda()
        self.timestep = 0


    def get_action(self, post: list[tuple[float, float, float]], left_img: list[PILImage], right_img: list[PILImage],
                   depth_img: list[PILImage]):
        """Returns a (dx, dy, dz) action prediction."""

        left_img = [self.transform(img) for img in left_img]
        right_img = [self.transform(img) for img in right_img]
        depth_img = [self.transform(img) for img in depth_img]

        left_img = torch.stack(left_img).unsqueeze(0)  # Shape becomes (1, 2, 3, H, W)
        right_img = torch.stack(right_img).unsqueeze(0)  # Shape becomes (1, 2, 3, H, W)
        depth_img = torch.stack(depth_img).unsqueeze(0)  # Shape becomes (1, 2, 1, H, W)

        # Create the positional tensor and add the batch dimension.
        post_tensor = torch.tensor(post, dtype=torch.float32)
        post_tensor = post_tensor.unsqueeze(0)  # Shape becomes (1, 2, 6)

        obs_dict = {
            'image_source_1': left_img,  # should be (1, 2, 3, H, W)
            'image_source_2': right_img,  # should be (1,2, 3, H, W)
            'depth': depth_img,  # should be (1, 2, 1, H, W)
            'agent_pos': post_tensor  # should be (1, 2, 3,)
        }
        print(right_img.shape)
        print(left_img.shape)
        print(post_tensor.shape)
        actions = self.model.predict_actions(obs_dict=obs_dict)
        #raw_action = actions.sum(dim=0).sum(dim=0).squeeze(0).cpu().detach().numpy()
        self.all_time_actions[[self.timestep], self.timestep:self.timestep + self.num_queries] = actions
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




class Subscribers:

    def __init__(self, update_every_ms):
        #TODO: remember to recalibrate this quaternion for every set of tests you run.
        self.camera_robot_transformation = [0.84248458,-0.03834644,-0.00607486,0.53630644]
        self.rotation = R.from_quat(self.camera_robot_transformation)
        self.rotation = self.rotation.as_matrix()
        self.rel_tip_pos = None
        self.update_every_ms = update_every_ms
        config = DiffusionConfig()
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(config=config)

        rospy.init_node('robot_test', anonymous=True)
        #rospy.Subscriber("/observation_state", Float32MultiArray, self.observation_callback)
        self.counter = 0
        self.rcm_publisher = rospy.Publisher('/ur5e_rcm_controller/set_rcm_pose', RcmPose, queue_size=0)
        self.GLOBAL_ROBOT_POS = None
        self.latest_state = None

        self.left_img_buffer = []
        self.right_img_buffer = []
        self.depth_img_buffer = []
        self.abs_tip_pos_buffer = []
        self.camera_tip_pos_buffer = []
        self.observation_buffer = None

        signal.signal(signal.SIGINT, self.shutdown)
        while True:
            self.update()
        #threading.Timer(2, self.update).start()
        #threading.Timer(update_every_ms, self.update).start()
        #rospy.spin()


    def publishData(self, position):
        MOVEMENT_TIME = 2
        msg = RcmPose()
        msg.targetPosition.x = position[0]
        msg.targetPosition.y = position[1]
        msg.targetPosition.z = position[2]
        msg.targetRoll = 0 #GLOBAL_ROLL
        msg.jerkLimit = 10
        msg.accelerationLimit = 5
        msg.epsilon = 0.00000001
        msg.movementTime = MOVEMENT_TIME
        rospy.loginfo(position)
        self.rcm_publisher.publish(msg)


    def move_robot(self, action):
        if abs(action[0]) > 0.01 or abs(action[1]) > 0.01 or abs(action[2]) > 0.01:
            rospy.loginfo("invalid action")
            return
        if self.GLOBAL_ROBOT_POS is None:
            self.GLOBAL_ROBOT_POS = self.rel_tip_pos
        action = np.dot(self.rotation.T, np.array(action)).tolist()
        self.GLOBAL_ROBOT_POS[0] += action[0]
        self.GLOBAL_ROBOT_POS[1] += action[1]
        self.GLOBAL_ROBOT_POS[2] += action[2]
        rospy.loginfo(msg=action)
        self.publishData(self.GLOBAL_ROBOT_POS)


    def convert_image(self, msg):
        # Get height and width directly from the message
        height = msg.height
        width = msg.width
        # Convert the byte data into a NumPy array
        img_data = np.frombuffer(msg.data, dtype=np.uint8)

        # Handle different image encodings
        if msg.encoding == 'rgb8':  # RGB Image (3 channels)
            img_data = img_data.reshape((height, width, 3))  # Reshape based on height and width
            img_data = img_data[..., ::-1]  # Convert BGR to RGB (if needed)
        elif msg.encoding == 'mono8':  # Grayscale Image (1 channel)
            img_data = img_data.reshape((height, width))  # Single channel for grayscale images
        else:
            raise ValueError(f"Unsupported encoding: {msg.encoding}")

        # Create a PIL image from the NumPy array
        if len(img_data.shape) == 3 and img_data.shape[2] == 3:  # RGB image
            pil_image = PILImage.fromarray(img_data, 'RGB')
        elif len(img_data.shape) == 2:  # Grayscale image (single channel)
            pil_image = PILImage.fromarray(img_data, 'L')
        else:
            raise ValueError(f"Unsupported image format with shape: {img_data.shape}")

        return pil_image

    def update(self):
        rospy.wait_for_service('get_observation_state')
        response = rospy.ServiceProxy('get_observation_state', ObservationState)
        #req = ObservationStateRequest()
        #response = (req)
        try:
            observation = response()
            #TODO: Update my local Depthcrafter service.
            left_frame = observation.left_frame
            right_frame = observation.right_frame
            depth_left = observation.depth_left
            robot_state = observation.robot_state
        except:
            raise ValueError("Depth service is not available.")
        if len(left_frame.data) == 0:
            rospy.loginfo("Message is empty.")
            #threading.Timer(0.005, self.update).start()
            return
        print(self.camera_tip_pos_buffer)

        relative_tip_pos = [robot_state.relativeTipPosition.x, robot_state.relativeTipPosition.y, robot_state.relativeTipPosition.z]
        self.rel_tip_pos = relative_tip_pos
        abs_tip_pos = [robot_state.tipPosition.x, robot_state.tipPosition.y, robot_state.tipPosition.z]
        camera_tip_pos = (np.array(abs_tip_pos) + np.dot(self.rotation, np.array(abs_tip_pos))).tolist()
        left_frame = self.convert_image(left_frame)
        right_frame = self.convert_image(right_frame)
        depth_left = self.convert_image(depth_left)

        left_frame.save(f"/mnt/cluster/temp/threading_il/current_observation/left_{self.counter}_left.png")
        depth_left.save(f"/mnt/cluster/temp/threading_il/current_observation/depth_{self.counter}_depth.png")
        right_frame.save(f"/mnt/cluster/temp/threading_il/current_observation/right_{self.counter}_right.png")
        self.left_img_buffer.append(left_frame)
        self.right_img_buffer.append(right_frame)
        self.depth_img_buffer.append(depth_left)
        self.abs_tip_pos_buffer.append(abs_tip_pos)
        self.camera_tip_pos_buffer.append(camera_tip_pos)
        if len(self.left_img_buffer) > 2:
            self.left_img_buffer = self.left_img_buffer[1:]
            self.right_img_buffer = self.right_img_buffer[1:]
            self.depth_img_buffer = self.depth_img_buffer[1:]
            self.abs_tip_pos_buffer = self.abs_tip_pos_buffer[1:]
            self.camera_tip_pos_buffer = self.camera_tip_pos_buffer[1:]

        self.counter += 1
        if self.counter>1 and len(self.left_img_buffer) > 1:
            action = self.model.get_action(post=self.camera_tip_pos_buffer, left_img=self.left_img_buffer,
                                           depth_img=self.depth_img_buffer, right_img=self.right_img_buffer)
            self.move_robot(action)

        #threading.Timer(0.7, self.update).start()


    '''def observation_callback(self, msg):
        if self.observation_buffer is None:
            self.observation_buffer = []
        data = msg.data
        self.observation_buffer.append(data)
        if len(self.observation_buffer) > 2:
            self.observation_buffer = self.observation_buffer[1:]
        if len(self.observation_buffer) > 1 :
            abs_tip_positions, rel_tip_positions, left_imgs, depth_imgs = unpack_observations(self.observation_buffer[-2:])
            action = self.model.get_action(abs_tip_positions, rel_tip_positions, left_imgs, depth_imgs)
            self.move_robot(action)



    def update(self):


        if (self.right_img is not None and self.left_img is not None and self.abs_tip_pos is not None and len(self.right_img) >1 and len(self.left_img) >1 and
                len(self.abs_tip_pos) >1):

            left_img = self.left_img[:]
            right_img = self.right_img[:]

            abs_tip_pos = [self.abs_tip_pos[0][:], self.abs_tip_pos[1][:]]
            abs_tip_pos_t = [
                np.dot(self.rotation, np.array(abs_tip_pos[0])).tolist(),
                np.dot(self.rotation, np.array(abs_tip_pos[1])).tolist()
            ]

            action = self.model.get_action(abs_tip_pos, abs_tip_pos_t, left_img, right_img)

            self.move_robot(action)
        threading.Timer(self.update_every_ms, self.update).start()'''


    def shutdown(self, signum, frame):
        rospy.signal_shutdown("Manual shutdown")  # Shutdown ROS
        sys.exit(0)  # Exit program


if __name__ == '__main__':
    x = Subscribers(0.01)
