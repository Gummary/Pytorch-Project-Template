"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse
from utils.logger import setup_logging
from configs.default import update_config
from configs import config



from agents import *

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--cfg',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    return parser.parse_args()
    

def main():
    global config
    args = parse_args()
    config = update_config(config, args)
    setup_logging(config)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.AGENT]
    agent = agent_class(config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
