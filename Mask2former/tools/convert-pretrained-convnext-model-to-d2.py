import torch

obj = torch.load('../checkpoints/convnextv2_huge_22k_512_ema.pt')['model']


new_state_dict = {}

for key, value in obj.items():
    new_key = 'backbone.' + key
    new_state_dict[new_key] = value

torch.save(new_state_dict, '../checkpoints/convnextv2_huge_22k_512_ema.pt')
