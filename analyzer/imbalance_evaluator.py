import math


class ImbalanceEvaluator:

    def __init__(self):
        pass

    def _get_distance_function(self, distance_function, empirical_distribution, uniform_distribution, m, k):
        if distance_function == 'Euclidean Distance':
            return self._euclidean_imbalance_degree(empirical_distribution, uniform_distribution, m, k)
        elif distance_function == 'Chebyshev Distance':
            return self._chebyshev_imbalance_degree(empirical_distribution, uniform_distribution, m, k)
        elif distance_function == 'Kullback Leibler Divergence':
            return self._kullback_leibler_imbalance_degree(empirical_distribution, uniform_distribution, m, k)
        elif distance_function == 'Hellinger Distance':
            return self._hellinger_imbalance_degree(empirical_distribution, uniform_distribution, m, k)
        elif distance_function == 'Total Variation Distance':
            return self._total_variation_imbalance_degree(empirical_distribution, uniform_distribution, m, k)
        elif distance_function == 'Chi Square Distance':
            return self._chi_square_imbalance_degree(empirical_distribution, uniform_distribution, m, k)
        else:
            raise ValueError(distance_function)

    def kullback_leibler_divergence(self, d1, d2):
        value = 0.0
        if len(d1) == len(d2):
            for i in range(0, len(d1)):
                if d1[i] == 0 or d2[i] == 0:
                    value += 0
                else:
                    value += d1[i] * math.log(d1[i]/d2[i])
            return value
        else:
            raise ValueError('d1 and d2 have different length!')

    def hellinger_distance(self, d1, d2):
        value = 0.0
        if len(d1) == len(d2):
            for i in range(len(d1)):
                value += math.pow(math.sqrt(d1[i])-math.sqrt(d2[i]), 2)
            value = 1 / math.sqrt(2) * math.sqrt(value)
            return value
        else:
            raise ValueError('d1 and d2 have different length!')

    def total_variation_distance(self, d1, d2):
        value = 0.0
        if len(d1) == len(d2):
            for i in range(0, len(d1)):
                value += abs(d1[i]-d2[i])
            return 0.5*value
        else:
            raise ValueError('d1 and d2 have different length')

    def chi_square_distance(self, d1, d2):
        value = 0.0
        if len(d1) == len(d2):
            for i in range(0, len(d1)):
                value += math.pow(d1[i]-d2[i], 2)/d2[i]
            return value
        else:
            raise ValueError('d1 and d2 have different length')

    def euclidean_distance(self, d1, d2):
        value = 0.0
        if len(d1) == len(d2):
            for i in range(0, len(d1)):
                value += math.pow(d1[i]-d2[i], 2)
            return math.sqrt(value)
        else:
            raise ValueError('d1 and d2 have different length')

    def chebyshev_distance(self, d1, d2):
        value = 0.0
        if len(d1) == len(d2):
            temp = list()
            for i in range(0, len(d1)):
                temp.append(abs(d1[i]-d2[i]))
            return max(temp)
        else:
            raise ValueError('d1 and d2 have different length')

    def _kullback_leibler_imbalance_degree(self, z, e, m, k):
        i = self.distribution_lowest_entropy_var(m, k)
        return self.kullback_leibler_divergence(z, e)/self.kullback_leibler_divergence(i, e) + (m-1)

    def _hellinger_imbalance_degree(self, z, e, m, k):
        i = self.distribution_lowest_entropy_var(m, k)
        return self.hellinger_distance(z, e)/self.hellinger_distance(i, e) + (m-1)

    def _total_variation_imbalance_degree(self, z, e, m, k):
        i = self.distribution_lowest_entropy_var(m, k)
        return self.total_variation_distance(z, e) / self.total_variation_distance(i, e) + (m - 1)

    def _chi_square_imbalance_degree(self, z, e, m, k):
        i = self.distribution_lowest_entropy_var(m, k)
        return self.chi_square_distance(z, e) / self.chi_square_distance(i, e) + (m - 1)

    def _euclidean_imbalance_degree(self, z, e, m, k):
        i = self.distribution_lowest_entropy_var(m, k)
        return self.euclidean_distance(z, e) / self.euclidean_distance(i, e) + (m - 1)

    def _chebyshev_imbalance_degree(self, z, e, m, k):
        i = self.distribution_lowest_entropy_var(m, k)
        return self.chebyshev_distance(z, e) / self.chebyshev_distance(i, e) + (m - 1)

    # The higher the value, the more imbalance exists between the two classes. 
    def get_imbalanced_ratio(self, z):
        return max(z) / min(z)

    def balanced_class_distribution(self, k):
        return [1./k for i in range(k)]

    def distribution_lowest_entropy_var(self, m, k):
        if m == 0:
            return self.balanced_class_distribution(k)
        else:
            distribution = list()
            for j in range(0, m):
                distribution.append(0)
            for j in range(0, k - m - 1):
                distribution.append(1 / k)
            distribution.append(1 - (k - m - 1) / k)
            return distribution
