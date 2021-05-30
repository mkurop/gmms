import collections
import copy
import random

import numpy as np
import pandas
import scipy.special


class GmmDiagonal:

    def __init__(self, train_set):
        self.train_set = np.asarray(train_set)

        if len(train_set) > len(train_set[0]):  # if training vectors as rows

            self.train_set = np.transpose(self.train_set)  # training vectors as columns

        self.__M = np.size(self.train_set, 1)  # number of training vectors

        self.__K = np.size(self.train_set, 1)  # number of components

        self.__d = np.size(self.train_set, 0)  # dimension of the training vectors

    """
    Computes log probability for given component and argument
    """

    def __log_gaussian(self, which_component, argument):
        diff = self.__mu[:, which_component] - argument

        log_prb = -.5 * (self.__d * np.log(2 * np.pi) + np.sum(np.log(self.__sigma[:, which_component])) +
                         np.sum(diff * diff / self.__sigma[:, which_component]))#tutaj zamiast dzielenia trzeba dać odwrotność

        return log_prb

    def __log_gaussian_vec(self, argument):

        diff = self.__mu - argument[:, np.newaxis]

        log_prb_vec = -.5 * (np.asarray([self.__d * np.log(2 * np.pi)]) + np.sum(np.log(self.__sigma), axis=0) + np.sum(
            diff * diff / self.__sigma, axis=0))

        return log_prb_vec

    def __log_gaussian_mat(self):

        # mu - points
        diff = self.__mu[np.newaxis, :, :].transpose((2, 0, 1)) - self.train_set[np.newaxis, :, :].transpose(
            (0, 2,
             1))  # results in KxMxd tensor, K - number of components, M - number of train points,
        # d - dimension of the train points

        # diff.T * sigma * diff
        under_exp_values = -.5 * np.einsum('kmi,ik,kmi->km', diff, self.__sigma ** (-1), diff)  # results in KxM matrix

        log_dets = -.5 * np.sum(np.log(self.__sigma), axis=0)

        log_values = -.5 * self.__d * np.log(2 * np.pi) + log_dets[:, np.newaxis] + under_exp_values

        return log_values

    """
    Computes log-likelihoods for all train vectors and all components, creates a matrix of size KxM
    also stores the log-likelihoods as an attribute for repeated use
    """

    def __log_prb_all(self):

        self.__log_liks = self.__log_gaussian_mat()

        return self.__log_liks

    def __log_likelihood_per_training_vector(self):
        log_liks = self.__log_prb_all()

        log_lik = np.ravel(scipy.special.logsumexp(log_liks + self.__log_pi)) / float(self.__M)  # multiply by proportions and normalize per training vector

        return log_lik

    def __initialize_pi(self):
        self.__log_pi = -np.ones((self.__K, 1)) * np.log(
            self.__K)  # initialize proportions (weights), create column vector

        self.__pi = np.exp(self.__log_pi)

    def __initialize(self):
        self.__mu = self.train_set  # initialize means of the Gaussian components

        self.__sigma = self.__initialize_sigma()

        self.__initialize_pi()  # initialize proportions (weights), create column vector

        self.__best_scores = np.ones((self.__K, 1)) * 1e20

        self.__best_mu = np.zeros((self.__d, self.__K))

        self.__best_sigma = np.zeros((self.__d, self.__K))

        self.__best_kernel_width = np.zeros((self.__K,))

    def __initialize_one_sigma(self, which_component):
        diff = self.train_set - np.transpose([self.__mu[:, which_component]])

        diff2 = diff * diff  # element-wise multiplication

        sigma = np.sum(diff2, axis=1)

        sigma /= np.float(self.__K - 1)  # -1 to account for the zero which_component column vector in diff

        sigma[sigma < 1e-16] = 1e-16  # for improvement of numerical stability of the algorithm

        return sigma

    def __initialize_sigma(self):

        diff = self.__mu[np.newaxis, :, :].transpose((2, 0, 1)) - self.train_set[np.newaxis, :, :].transpose(
            (0, 2, 1))

        diff2 = diff * diff

        sigma = np.einsum('kmi->ik', diff2)

        sigma /= np.float(self.__K - 1)

        sigma[sigma < 1e-16] = 1e-16

        return sigma

    def __update_log_responsibilities(self):

        self.__log_responsibilities = self.__log_responsibility_mat()

        return self.__log_responsibilities

    def __log_responsibility(self, which_component, which_vector):
        aux = self.__log_liks[:, which_vector] + self.__log_pi

        log_resp = aux[which_component] - scipy.special.logsumexp(aux)

        # resp = np.exp(aux[which_component] - scipy.special.logsumexp(aux))

        return log_resp

    def __log_responsibility_vec(self, which_vector):
        aux = self.__log_liks[:, which_vector] + np.ravel(self.__log_pi)

        log_resp = aux - np.ravel(scipy.special.logsumexp(aux))

        return log_resp

    def __log_responsibility_mat(self):

        aux = self.__log_liks + self.__log_pi

        log_resp = aux - np.ravel(scipy.special.logsumexp(aux, axis=0))[np.newaxis, :]

        return log_resp

    def __update_log_nk(self):

        self.__log_nk = scipy.special.logsumexp(self.__log_responsibilities, axis=1)

        return self.__log_nk

    def __update_log_n(self):

        self.__log_n = scipy.special.logsumexp(self.__log_nk)

        return self.__log_n

    def __update_mu(self):

        self.__mu = self.__mu_all()

    def __mu_one(self, which_component):

        mu = np.sum(
            np.exp(self.__log_responsibilities[which_component, :] - self.__log_nk[which_component]) * self.train_set,
            axis=1)

        return mu

    def __mu_all(self):

        aux = np.exp(self.__log_responsibilities - np.ravel(self.__log_nk)[:, np.newaxis])

        mu = np.einsum('km, dm -> dk', aux, self.train_set)

        return mu

    def __update_sigma(self):

        self.__sigma = self.__sigma_all()

    def __sigma_one(self, which_component):

        diff = self.train_set - np.transpose([self.__mu[:, which_component]])

        sigma = np.sum(
            np.exp(self.__log_responsibilities[which_component, :] - self.__log_nk[which_component]) * diff * diff,
            axis=1)

        sigma[sigma < 1e-16] = 1e-16  # for improvement of the numerical stability of the algorithm

        return sigma

    def __sigma_all(self):

        diff = self.__mu[np.newaxis, :, :].transpose((2, 0, 1)) - self.train_set[np.newaxis, :, :].transpose(
            (0, 2, 1))

        diff2 = diff * diff

        aux = np.exp(self.__log_responsibilities - np.ravel(self.__log_nk)[:, np.newaxis])

        sigma = np.einsum('km, kmi -> ik', aux, diff2)

        sigma[sigma < 1e-16] = 1e-16  # for improvement of the numerical stability of the algorithm

        return sigma

    def __update_log_pi(self):

        self.__log_pi = self.__log_nk - np.asarray(self.__log_n)

    def __update_pi(self):

        self.__update_log_pi()

        self.__pi = np.exp(self.__log_pi)

    def __e_step(self):

        self.__update_log_responsibilities()

    def __m_step(self):

        self.__update_log_nk()

        self.__update_log_n()

        self.__update_mu()  # update means ---> changes 17.03.2020

        self.__update_sigma()  # update diagonal covariance matrices (column variances vectors)

        self.__update_pi()  # update weights

    def __scores(self):

        self.__scores = np.zeros((self.__K, 1))

        for k in range(self.__K):
            self.__scores[k] = self.__score(k)

    """
    Score is a differential entropy of the component plus coding loss due to mismatch between true and estimated 
    component covariance matrix (score is measured in nats)
    """

    def __score(self, which_component):
        kw = self.__effective_kernel_width(which_component)

        score = self.__entropy(which_component) + self.__expected_kullback_leibler_divergence(kw)

        return score

    def __effective_kernel_width(self, which_component):
        aux = self.__log_liks[which_component, :]

        p = aux - scipy.special.logsumexp(aux)

        e = -np.sum(np.exp(p) * p)  # entropy

        return np.exp(e)  # return effective kernel width in samples (may be fractional)

    def __effective_kernel_width_all(self):

        p = self.__log_liks - np.ravel(scipy.special.logsumexp(self.__log_liks, axis=1))[:, np.newaxis]

        e = -np.sum(np.exp(p) * p, axis=1)

        return np.exp(e)

    """
    Computes entropy in nats for a given gaussian mixture component
    """

    def __entropy(self, which_component):
        h = .5 * np.sum(np.log(2 * np.pi * np.e * self.__sigma[:, which_component]))

        return h

    """
    Computes the KL divergence in nats according to a formula developed in the paper "Expected Kullback-Leibler 
    Divergence for Multivariate Gaussians" (which is attached to the project)
    """

    def __expected_kullback_leibler_divergence(self, kernel_width):
        kl = .5 * self.__d * (scipy.special.psi(.5 * max(kernel_width - 1, 1)) + np.log(2) - np.log(
            max(kernel_width - 1, 1e-16)) - 1 + kernel_width / max(kernel_width - 2, 1e-16) + (kernel_width - 1) / (
                                      max(kernel_width, 1e-16) * max(kernel_width - 2, 1e-16)))

        return kl

    def fit(self, number_of_iterations=300):

        self.__initialize()

        log_lik = self.__log_likelihood_per_training_vector()

        i = 0

        print(f"Log likelihood per training vector at iteration {i} is equal {log_lik[0]}")

        scores = []

        for i in range(1, number_of_iterations):
            self.__e_step()

            self.__m_step()

            log_lik = self.__log_likelihood_per_training_vector()

            kernel_width: np.array = self.__effective_kernel_width_all()

            for k in range(self.__K):

                s = self.__score(k)

                if s < self.__best_scores[k]:
                    self.__best_scores[k] = s

                    self.__best_mu[:, k] = copy.copy(self.__mu[:, k])

                    self.__best_sigma[:, k] = copy.copy(self.__sigma[:, k])

                    self.__best_kernel_width[k] = kernel_width[k]

            print(f"Log likelihood per training vector at iteration {i} is equal {log_lik[0]}")

            print(f"Entropy of the 1st component {self.__entropy(1)}")
            #
            print(f"Effective kernel width {self.__effective_kernel_width(1)}")
            #
            print(f"Score per 1st component {self.__score(1)}")
            #
            print(f"KL distance {self.__expected_kullback_leibler_divergence(self.__effective_kernel_width(1))}")

            scores.append(self.__score(1))

        self.__mu = self.__best_mu
        self.__sigma = self.__best_sigma

        self.__initialize_pi()
        for i in range(4):
            log_lik = self.__log_likelihood_per_training_vector()

            self.__update_log_responsibilities()

            self.__update_log_nk()

            self.__update_log_n()

            self.__update_pi()
            print(f"Final log_lik {log_lik}")

    def get_gmm(self):
        gmm = {'mu': self.__best_mu, 'sigma': self.__best_sigma, 'sigma_inv': 1 / self.__best_sigma,
               'log_pi': self.__log_pi}
        return gmm


def get_fields_subset(fields, which):
    return [field for field in fields if field in set(which)]


def evaluate_each_diagonal_component_(gmm_, points_: np.array):
    assert np.size(points_, axis=0) == np.size(gmm_['mu'], axis=0)

    d = np.size(points_, axis=0)

    # mu - points
    diff = gmm_['mu'][np.newaxis, :, :].transpose((2, 0, 1)) - points_[np.newaxis, :, :].transpose(
        (0, 2,
         1))  # results in KxMxd tensor, K - number of components, M - number of train points,
    # d - dimension of the train points

    # diff.T * sigma * diff
    under_exp_values = -.5 * np.einsum('kmi,ik,kmi->km', diff, gmm_['sigma_inv'], diff)  # results in KxM matrix

    log_dets = -.5 * np.sum(np.log(gmm_['sigma']), axis=0)

    log_values = -.5 * d * np.log(2 * np.pi) + log_dets[:, np.newaxis] + under_exp_values

    return np.exp(log_values), log_values


def predictor(gmm, predicted_variables_indices, predictor_values):
    """
    Predictor function.
    predicted_variables_indices - list of predicted variables, zero based
    """

    d = gmm['mu'].shape[0]

    # get predictor indices
    predictor_variables_indices = sorted(list(set(range(d)).difference(set(predicted_variables_indices))))

    # means for predicted variables
    mu_predicted = np.take(gmm['mu'], predicted_variables_indices, axis=0)

    assert len(predictor_variables_indices) == len(predictor_values)

    # form gmm for predictor variables
    if predictor_variables_indices:
        mu_predictor = np.take(gmm['mu'], predictor_variables_indices, axis=0)
        sigma_predictor = np.take(gmm['sigma'], predicted_variables_indices, axis=0)
        sigma_inv_predictor = np.take(gmm['sigma_inv'], predicted_variables_indices, axis=0)
        gmm_predictor = {'mu': mu_predictor, 'sigma': sigma_predictor, 'sigma_inv': sigma_inv_predictor,
                         'log_pi': gmm['log_pi']}

        # evaluate gmm_predictor for predictor values
        liks, log_liks = evaluate_each_diagonal_component_(gmm_predictor, predictor_values)

        log_liks += gmm_predictor['log_pi'][np.newaxis, :]

        # evaluate predicted value
        predicted = mu_predicted * np.exp(log_liks - scipy.special.logsumexp(log_liks))[np.newaxis, :]
    else:
        predicted = mu_predicted * np.exp(gmm['log_pi'])[np.newaxis, :]

    return predicted


def evaluate_on_test_set(data_test, gmms, predictor_variables, categorical_fields, continuous_fields):
    table_categorical = get_subtable(get_fields_subset(predictor_variables, categorical_fields), data_test)
    table_continuous = get_subtable(get_fields_subset(predictor_variables, continuous_fields), data_test)

    predicted_values_loss = [
        (predictor(gmms[tuple(row_categorical)], [0], np.asarray(table_continuous[i][1:])) - table_continuous[i][
            0]) ** 2
        for
        i, row_categorical in
        enumerate(table_categorical) if tuple(row_categorical) in gmms]

    return np.mean(np.asarray(predicted_values_loss))


def train(train_sets):
    gmms = {}

    for cat_tuple, train_set in train_sets.items():
        print(
            f"Start training for {cat_tuple}, dimension of the training set {len(train_set[0])}, size of the training "
            f"set {len(train_set)}")
        gmm = GmmDiagonal(train_set)

        gmm.fit()

        gmms[cat_tuple] = gmm.get_gmm()

    return gmms


def get_subtable(fields, data):
    """
    Function gets a subtable of the data
    Returned is list of lists in the row major format
    """

    cols = []

    for field in fields:
        cols.append(data[field].tolist())

    return list(map(list, zip(*cols)))


def get_train_sets(data, fields, categorical_fields, continuous_fields):
    train_sets = collections.defaultdict(list)

    cat_rows = get_subtable(get_fields_subset(fields, categorical_fields), data)

    con_rows = get_subtable(get_fields_subset(fields, continuous_fields), data)

    # value_rows = get_subtable(['value'], data)

    # if len(cat_rows) != len(con_rows):
    #     print(f"cat_rows = {len(cat_rows)}, con_rows = {len(con_rows)}")
    #     input()

    if not cat_rows:
        train_sets[()] = con_rows

        return train_sets

    for i, cat_row in enumerate(cat_rows):
        train_sets[tuple(cat_row)].append(con_rows[i])

    return train_sets


if __name__ == "__main__":

    # read data

    data = pandas.read_excel('mydata.xls', sheet_name='data')

    # create train and test tests

    test_rows = random.choices(list(range(data.shape[0])), k=round(.1 * data.shape[0]))

    train_rows = list(set(range(data.shape[0])).difference(set(test_rows)))

    data_test = data.iloc[test_rows, :]

    data_train = data.iloc[train_rows, :]

    # read schema

    data_schema = pandas.read_excel('mydata.xls', sheet_name='schema')

    # read fields names

    header_names = data.columns.ravel()

    # skip non-characteristics

    header_names = header_names[4:]

    # create list of categorical and continuous data fields

    categorical_fields = []
    continuous_fields = []
    for i, data_field in enumerate(data_schema['Field name'][4:]):
        # print(i)
        if data_schema['categorical'][i + 4] == 1:
            categorical_fields.append(data_field)
        else:
            continuous_fields.append(data_field)

    # checking the sanity of data

    # all_fields = set(header_names)
    # sum_cat_cont = set(categorical_fields).union(set(continuous_fields))
    # print(f"{len(all_fields)}, {len(sum_cat_cont)}")

    # count categorical variables (the cartesian product)

    dict_of_categorical = {}
    count = 1
    for field in categorical_fields:
        dict_of_categorical[field] = list(set(data[field]))
        # print(dict_of_categorical[field])
        count *= len(dict_of_categorical[field])

    print(f"Cardinality of the cartesian product of categorical variables: {count}")

    # compute the diversity index

    table_categorical = get_subtable(categorical_fields, data_train)

    unique_categorical = set(map(tuple, table_categorical))

    # diversity index

    Z = len(unique_categorical)

    print(f"Diversity index for categorical variables: {Z}")

    # generalization coefficient / training ratio

    N = float(len(table_categorical)) / len(unique_categorical)

    print(f"Generalization coefficient for categorical variables: {N}")

    # probability of unseen events

    PrA = 1 / N ** 2

    print(f"Probability of unseen events for categorical variables: {PrA}")

    train_sets = get_train_sets(data_train, header_names, categorical_fields, continuous_fields)

    # skeleton for training

    characteristics_fields = list(header_names[1:])

    current_set = characteristics_fields

    prev_best_predictive = ['value']

    for i in range(len(characteristics_fields)):

        print(f"Processing field number i = {i}")

        best_score = 0

        for field in current_set:

            # form predictive

            predictive = copy.copy(prev_best_predictive)

            predictive.append(field)

            # form training sets

            training_sets = get_train_sets(data, predictive, categorical_fields, continuous_fields)

            # train model for predictive

            gmms = train(training_sets)

            # asses predictive

            test_loos = evaluate_on_test_set(data_test, gmms, predictive, categorical_fields, continuous_fields)

            score = test_loos

            if score > best_score:
                best_field: object = field

                best_predictive: object = predictive

        current_set.remove(best_field)

        prev_best_predictive = best_predictive

    print(prev_best_predictive)

    for k, v in train_sets.items():
        print(k)
        print(v[:2])

    sub = get_fields_subset(header_names, categorical_fields)

    print(sub)

    input()

    print(data_schema['categorical'])

    print(data)

    print(header_names)

    # test set
