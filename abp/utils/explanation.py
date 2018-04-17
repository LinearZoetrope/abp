import numpy as np
import torch
from torch.autograd import Variable


from abp.models import HRAModel

import excitationbp as eb

class Explanation(object):
    """ Explanation for an Adaptive Variable """
    #TODO currenly supports only HRA Adaptive

    def __init__(self, adaptive):
        super(Explanation, self).__init__()
        self.adaptive = adaptive


    def pdx(self, q_values, selected_action, target_actions):
        _pdx = [
                [q_values[r][selected_action] - q_values[r][target]  for target in target_actions]
                for r in range(len(q_values))
                ]
        return np.array(_pdx)

    def generate_pdx(self, q_values):
        pdx = {}
        n_actions = len(q_values[0])
        for i in range(n_actions):
            for j in range(n_actions):
                if i != j:
                    pdx[(i, j)] = self.pdx(q_values, i, [j])
        return pdx

    def generate_saliencies(self, state):
        eb.use_eb(True)

        state = Variable(torch.Tensor(state)).unsqueeze(0)

        saliencies = {}
        for idx, choice in enumerate(self.adaptive.choices):
            choice_saliencies = {}
            prob_outputs = Variable(torch.zeros((len(self.adaptive.choices),)))
            prob_outputs[idx] = 1

            for reward_idx, reward_type in enumerate(self.adaptive.reward_types):
                #TODO better way to do this
                explainable_model = HRAModel(self.adaptive.name + "_explain", self.adaptive.network_config, restore = False)
                explainable_model.replace(self.adaptive.eval_model)

                explainable_model.clear_weights(reward_idx)

                layer_top = explainable_model.top_layer(reward_idx)

                saliency = eb.excitation_backprop(explainable_model.model, state, prob_outputs, contrastive = False, layer_top = layer_top, target_layer = 0)

                choice_saliencies[reward_type] = np.squeeze(saliency.view(*state.shape).data.numpy())

            # for overall reward
            saliency = eb.excitation_backprop(self.adaptive.eval_model.model, state, prob_outputs, contrastive = False, target_layer = 0)
            choice_saliencies["all"] = np.squeeze(saliency.view(*state.shape).data.numpy())

            saliencies[choice] = choice_saliencies

        eb.use_eb(False)
        return saliencies
