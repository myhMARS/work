from id3 import entropy, conditional_entropy


def info_gain(probabilities, marginal_probabilities, joint_proabilities):
    return entropy(probabilities) - conditional_entropy(marginal_probabilities, joint_proabilities)


def info_gain_ratio(probabilities, marginal_probabilities, joint_probabilities):
    info_gain_value = info_gain(probabilities, marginal_probabilities, joint_probabilities)
    ratio = info_gain_value / entropy(marginal_probabilities)
    return ratio


if __name__ == "__main__":
    probabilities = [0.5, 0.5]
    marginal_probabilities = [5 / 14.0, 4 / 14.0, 5 / 14.0]
    joint_probabilities = [
        [(5 / 14.0) * 0.6, (5 / 14.0) * 0.4],
        [(4 / 14.0) * 0, (4 / 14.0) * 1.0],
        [(5 / 14.0) * 0.8, (5 / 14.0) * 0.2]
    ]
    # print(info_gain(probabilities, marginal_probabilities, joint_probabilities))
    print(info_gain_ratio(probabilities, marginal_probabilities, joint_probabilities))
