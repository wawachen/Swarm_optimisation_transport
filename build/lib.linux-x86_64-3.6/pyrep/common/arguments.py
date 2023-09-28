import argparse

"""
Here are the params for the transport

"""

def get_args():
    parser = argparse.ArgumentParser("Load transport experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="cooperative_transport", help="name of the scenario script")
    parser.add_argument("--kp", type=float, default=15.0, help="transport controller")
    args = parser.parse_args()

    return args
