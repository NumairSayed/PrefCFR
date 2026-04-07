import numpy as np

game_config = {
    "kuhn_poker"    : {},
    "leduc_poker"  : {},
}

pref_config = {
    "kuhn_poker"    : {
        "normal"   : {},
        "offensive5": {
            "0": [
                np.array([1, 5]),
                0
            ],
            "1": [
                np.array([1, 5]),
                0
            ],
            "2": [
                np.array([1, 5]),
                0
            ]
        },
        "offensive10": {
            "0": [
                np.array([1, 10]),
                0
            ],
            "1": [
                np.array([1, 10]),
                0
            ],
            "2": [
                np.array([1, 10]),
                0
            ]
        },
        "defensive5": {
            "0": [
                np.array([5, 1]),
                0
            ],
            "1": [
                np.array([5, 1]),
                0
            ],
            "2": [
                np.array([5, 1]),
                0
            ]
        },
        "defensive10": {
            "0": [
                np.array([10, 1]),
                0
            ],
            "1": [
                np.array([10, 1]),
                0
            ],
            "2": [
                np.array([10, 1]),
                0
            ]
        }

    },
    "leduc_poker"   : {
        "normal"         : {},
        "offensive_10_05": {
            '[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 10]),
                0.05
            ],
            '[Observer: 0][Private: 1][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 10]),
                0.05
            ],
            '[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 10]),
                0.05
            ],
            '[Observer: 0][Private: 3][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 10]),
                0.05
            ],
            '[Observer: 0][Private: 4][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 10]),
                0.05
            ],
            '[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 10]),
                0.05
            ],
        },
        "offensive_10_00": {
            '[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 10]),
                0.00
            ],
            '[Observer: 0][Private: 1][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 10]),
                0.00
            ],
            '[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 10]),
                0.00
            ],
            '[Observer: 0][Private: 3][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 10]),
                0.00
            ],
            '[Observer: 0][Private: 4][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 10]),
                0.00
            ],
            '[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 10]),
                0.00
            ],
        },
        "offensive_5_05" : {
            '[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.05
            ],
            '[Observer: 0][Private: 1][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.05
            ],
            '[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.05
            ],
            '[Observer: 0][Private: 3][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.05
            ],
            '[Observer: 0][Private: 4][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.05
            ],
            '[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.05
            ],
        },

        "offensive_5_10" : {
            '[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.10
            ],
            '[Observer: 0][Private: 1][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.10
            ],
            '[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.10
            ],
            '[Observer: 0][Private: 3][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.10
            ],
            '[Observer: 0][Private: 4][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.10
            ],
            '[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.10
            ],
        },

        "offensive_5_20" : {
            '[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.20
            ],
            '[Observer: 0][Private: 1][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.20
            ],
            '[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.20
            ],
            '[Observer: 0][Private: 3][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.20
            ],
            '[Observer: 0][Private: 4][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.20
            ],
            '[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.20
            ],
        },

        "offensive_5_50" : {
            '[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.50
            ],
            '[Observer: 0][Private: 1][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.50
            ],
            '[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.50
            ],
            '[Observer: 0][Private: 3][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.50
            ],
            '[Observer: 0][Private: 4][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.50
            ],
            '[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([1, 5]),
                0.50
            ],
        },

        "defensive_10_05": {
            '[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([10, 1]),
                0.05
            ],
            '[Observer: 0][Private: 1][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([10, 1]),
                0.05
            ],
            '[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([10, 1]),
                0.05
            ],
            '[Observer: 0][Private: 3][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([10, 1]),
                0.05
            ],
            '[Observer: 0][Private: 4][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([10, 1]),
                0.05
            ],
            '[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([10, 1]),
                0.05
            ],
        },
        "defensive_10_00": {
            '[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([10, 1]),
                0.00
            ],
            '[Observer: 0][Private: 1][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([10, 1]),
                0.00
            ],
            '[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([10, 1]),
                0.00
            ],
            '[Observer: 0][Private: 3][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([10, 1]),
                0.00
            ],
            '[Observer: 0][Private: 4][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([10, 1]),
                0.00
            ],
            '[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([10, 1]),
                0.00
            ],
        },
        "defensive_5_05" : {
            '[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([5, 1]),
                0.05
            ],
            '[Observer: 0][Private: 1][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([5, 1]),
                0.05
            ],
            '[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([5, 1]),
                0.05
            ],
            '[Observer: 0][Private: 3][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([5, 1]),
                0.05
            ],
            '[Observer: 0][Private: 4][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([5, 1]),
                0.05
            ],
            '[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]': [
                np.array([5, 1]),
                0.05
            ],
        }

    },
}
