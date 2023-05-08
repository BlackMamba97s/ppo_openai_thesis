import gym
import numpy as np

from config import args

class TestEnvironment():
    """
    Test environment wrapper for CarRacing, used to test PPO.
    There is no actual difference between the environment in testing and training.
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(args.seed)

        # Get the reward threshold for the environment
        self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        """
        Reset the environment and return the initial state.

        Returns:
            The initial state of the environment as a numpy array of stacked grayscale images.
        """
        # Reset the episode counter and the reward memory
        self.counter = 0
        self.av_r = self.reward_memory()

        # Reset the environment and get the initial RGB image
        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb) #convert into grayscale, see thesis
        self.stack = [img_gray] * args.img_stack # stack as initial state
        return np.array(self.stack)

    def step(self, action):
        """
        Take an action in the environment and return the resulting state, reward, and other information.

        Args:
            action: The action to take in the environment.

        Returns:
            A tuple of (next_state, reward, done, die), where
            - next_state: The next state of the environment as a numpy array of stacked grayscale images.
            - reward: The reward for taking the given action in the current state.
            - done: A boolean indicating whether the episode has ended due to reaching the maximum number of steps.
            - die: A boolean indicating whether the car has crashed in the current state.
        """
        total_reward = 0
        for i in range(args.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == args.img_stack
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        """
        Render the current state of the environment.

        Args:
            *arg: Additional arguments to pass to the render method of the underlying environment.
        """
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        """
        Convert an RGB image to grayscale.

        Args:
            rgb: The RGB image to convert.
            norm: Whether to normalize the resulting grayscale image.

        Returns:
            The grayscale image as a numpy array.
        """
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        """
        Create a reward memory function to track the average reward over the past N steps.

        Returns:
            A function that takes a reward as input and returns the average reward over the past N steps.
        """
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory