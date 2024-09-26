import pandas as pd, numpy as np, math

# Play Tennis dataset
data = {'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']}
df = pd.DataFrame(data)

# Entropy calculation
def entropy(col): return -sum([(p/len(col)) * math.log2(p/len(col)) for p in np.unique(col, return_counts=True)[1]])

# ID3 algorithm
def id3(data, features, target='PlayTennis'):
    if len(np.unique(data[target])) == 1: return data[target].iloc[0]
    if not features: return data[target].mode()[0]
    gains = [(entropy(data[target]) - sum((len(subset) / len(data)) * entropy(subset[target]) for subset in [data[data[f] == v] for v in np.unique(data[f])])) for f in features]
    best_feature = features[np.argmax(gains)]
    tree = {best_feature: {v: id3(data[data[best_feature] == v], [f for f in features if f != best_feature], target) for v in np.unique(data[best_feature])}}
    return tree

# Build decision tree and classify new sample
decision_tree = id3(df, ['Outlook', 'Temperature', 'Humidity', 'Wind'])
new_sample = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}

def classify(tree, sample):
    if not isinstance(tree, dict): return tree
    attribute = list(tree.keys())[0]
    return classify(tree[attribute][sample[attribute]], sample)

print("Decision Tree:", decision_tree)
print("New Sample Classification:", classify(decision_tree, new_sample))