SUBJECTS = [1, 2, 3, 4, 5, 6]
MODES = ["train", "predict", "all"]
TRANSFORMERS = ["CSP", "SPoC"]

EXPERIMENTS = {
    "hands_vs_feet__imagery": ['imagine/hands', 'imagine/feet'],
    "hands_vs_feet__action": ['do/hands', 'do/feet'],
    "imagery_vs_action__feets": ['do/feet', 'imagine/feet'],
    "imagery_vs_action__hands": ['do/hands', 'imagine/hands'],
}

EXPERIMENTS_IDS = {
    'imagery': [6, 10, 14],
    'action': [5, 9, 13],
}
