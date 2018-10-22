from abp.configs import NetworkConfig
from abp.models import HRAModel
from abp.utils import clear_summary_path

from tensorboardX import SummaryWriter

import numpy as np
import numpy.random as rand

import logging
logger = logging.getLogger('root')


class PolicyIter(object):

    def __init__(self, name, choices, reward_types, network_config, reinforce_config):
        import sys
        super(PolicyIter, self).__init__()

        self.name = name
        self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config
        self.replace_frequency = reinforce_config.replace_frequency

        target_model = NetworkConfig.load_from_yaml(
            network_config.target_model)
        # Let's not use CUDA, it makes saliency difficult
        self.target_model = HRAModel(
            self.name + "_target", target_model, False)
        self.model = HRAModel(self.name, self.network_config, False)

        self.learning = True
        self.reward_types = reward_types
        self.steps = 0
        self.episode = 0
        self.reward_history = []
        self.best_reward_mean = -sys.maxsize

        reinforce_summary_path = self.reinforce_config.summaries_path + "/" + self.name

        if not network_config.restore_network:
            clear_summary_path(reinforce_summary_path)
        else:
            self.restore_state()

        self.summary = SummaryWriter(log_dir=reinforce_summary_path)

        self.chosen_random = False
        self.approx_steps_per_ep = reinforce_config.approx_steps_per_ep
        self.random_step = rand.randint(self.approx_steps_per_ep)

    def restore_state(self):
        import cPickle as pickle
        restore_path = self.network_config.network_path + "/adaptive.info"

        if self.network_config.network_path and os.path.exists(restore_path):
            logger.info("Restoring state from %s" %
                        self.network_config.network_path)

            with open(restore_path, "rb") as file:
                info = pickle.load(file)

            self.steps = info["steps"]
            self.best_reward_mean = info["best_reward_mean"]
            self.episode = info["episode"]

            logger.info("Continuing from %d episode (%d steps) with best reward mean %.2f" %
                        (self.episode, self.steps, self.best_reward_mean))

    def save(self, force=False):
        import cPickle as pickle
        info = {
            "steps": self.steps,
            "best_reward_mean": self.best_reward_mean,
            "episode": self.episode
        }

        if force:
            logger.info("Forced to save network")
            self.model.save_network()
            pickle.dump(info, self.network_config.network_path +
                        "adaptive.info")

        if (len(self.reward_history) >= self.network_config.save_steps and
                self.episode % self.network_config.save_steps == 0):

            total_reward = sum(
                self.reward_history[-self.network_config.save_steps:])
            current_reward_mean = total_reward / self.network_config.save_steps

            if True:  # current_reward_mean >= self.best_reward_mean:
                self.best_reward_mean = current_reward_mean
                info["best_reward_mean"] = current_reward_mean
                logger.info(
                    "Saving network. Found new best reward (%.2f)" % current_reward_mean)
                self.model.save_network()
                with open(self.network_config.network_path + "/adaptive.info", "wb") as file:
                    pickle.dump(info, file, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                logger.info("The best reward is still %.2f. Not saving" %
                            current_reward_mean)
