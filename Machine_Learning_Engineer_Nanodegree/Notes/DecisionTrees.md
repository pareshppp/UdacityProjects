# Decision Trees

### Entropy
$$Entropy = \sum_i -P_i.log(P_i)$$ 
$$i = class\ within\ variable\ v$$ 
$$P_i = Probability\ (fraction)\ of\ class\ i\ within\ variable\ v$$
- Entropy is the measure of randomness in a variable.
- Entropy is the measure of impurity in a variable.
- 100% randomness (50-50 split) --> Maximum Impurity --> Entropy = 1 
- 0% randomness (perfect classification) --> Zero Impurity --> Entropy = 0 

### Information Gain
$$Information Gain = Entropy(Parent) - [Weighted Average].Entropy(Children)$$
- DT uses Information Gain to make splits.
- Every split is made so that the IG is maximised.
