import torch
# from transformers import DataCollatorWithPadding

class PolicyDataCollator():

    def __call__(self, features):
        states = torch.cat([f['states'] for f in features], dim=0)
        actions = torch.cat([f['actions'] for f in features], dim=0)
        
        return {
            'states': torch.tensor(states),
            'actions': torch.tensor(actions),
        }