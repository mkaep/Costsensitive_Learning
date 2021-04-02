import collections
from analyzer import imbalance_evaluator


def calculate_imbalance_degree(labels):
    k = len(set(labels))
    n = len(labels)

    balanced_distribution = [1./k for i in range(k)]
    empirical_distribution = list()

    c = collections.Counter(labels)
    for label in c.keys():
        empirical_distribution.append(c[label]/n)

    m = 0
    for i in empirical_distribution:
        if i < 1./k:
            m = m+1

    print(empirical_distribution)
    print('Number of minority classes: ', m)
    print('Number of classes: ', k)
    print('Number of elements: ', n)
    print('Threshold: ', 1/k)
    # Count number of majority classes

    uniform_distribution = [1./k for i in range(k)]
    imbEva = imbalance_evaluator.ImbalanceEvaluator()
    print("EU: ", imbEva._get_distance_function('Euclidean Distance', empirical_distribution, uniform_distribution, m, k))
    print("KL: ",
          imbEva._get_distance_function('Kullback Leibler Divergence', empirical_distribution, uniform_distribution, m, k))
    print("HE: ",
          imbEva._get_distance_function('Hellinger Distance', empirical_distribution, uniform_distribution, m, k))
    print("TV: ",
          imbEva._get_distance_function('Total Variation Distance', empirical_distribution, uniform_distribution, m, k))
    print("CS: ",
          imbEva._get_distance_function('Chi Square Distance', empirical_distribution, uniform_distribution, m, k))
    print("CH: ",
          imbEva._get_distance_function('Chebyshev Distance', empirical_distribution, uniform_distribution, m, k))
    print("IR: ", imbEva.get_imbalanced_ratio(empirical_distribution))