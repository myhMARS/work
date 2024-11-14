import math

def entropy(probabilities):
    entropy_value = 0
    for p in probabilities:
        if p > 0:
            entropy_value -= p * math.log2(p)
    return entropy_value

def conditional_entropy(marginal_probabilities, joint_probabilities):
    num_x = len(marginal_probabilities)
    num_y = len(joint_probabilities[0])
    condition_entropy = 0
    
    for i in range(num_x):
        conditional_probs_y_given_x = []
        for j in range(num_y):
            p_y_given_x = joint_probabilities[i][j] / marginal_probabilities[i] if marginal_probabilities[i] > 0 else 0
            conditional_probs_y_given_x.append(p_y_given_x)
        print("condition:",entropy(conditional_probs_y_given_x))
        condition_entropy += marginal_probabilities[i] * entropy(conditional_probs_y_given_x)

    return condition_entropy

if __name__ == "__main__":
    probabilities = [0.5, 0.5]
    print(entropy(probabilities))


    marginal_probabilities = [5/14.0, 4/14.0, 5/14.0]
    joint_probabilities = [
        [(5/14.0)*0.6,(5/14.0)*0.4],
        [(4/14.0)*0,(4/14.0)*1.0],
        [(5/14.0)*0.8,(5/14.0)*0.2]
    ]
    conditional_entropy(marginal_probabilities,joint_probabilities)

