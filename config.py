import argparse
import torch

# Argument parsing for configuration settings
parser = argparse.ArgumentParser(description='Train or Test Open AI Gym Car Racing with PPO')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test the agent')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 12)')
parser.add_argument('--file-name', type=str, default='ppo_net_params.pkl', help='file name to load or save the model parameters')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--render', action='store_true', help='render the environment')

args = parser.parse_args()


# Check for CUDA availability and set the device accordingly
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed(args.seed)