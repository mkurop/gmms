import collections
import copy

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

class GmmFull:

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

    def __log_gaussian_mat(self):
        #tu był duży błąd ----- liczyłem bez sensu logarytm macierzowy
        # mu - points
        diff = self.__mu[np.newaxis, :, :].transpose((2, 0, 1)) - self.train_set[np.newaxis, :, :].transpose((0, 2,1))
        # results in KxMxd tensor, K - number of components, M - number of train points,
        # d - dimension of the train points

        # diff.T * sigma * diff
        #under_exp_values = -.5 * np.einsum('kmi,ik,kmi->km', diff, self.__sigma ** (-1), diff)  # results in KxM matrix
        under_exp_values = -.5 * np.einsum('kmi,kij,kmj->km', diff, np.linalg.inv(self.__sigma), diff)  # results in KxM matrix

        #w,v = np.linalg.eig(self.__sigma)
        #diago_log_w = np.einsum('ij,li->lji', np.eye(2), np.log(np.real(w))) #tworzenie macierzy diagonalnych z wektorów wartości własnych

        #log_sigma = np.einsum('lij,ljk,lmk->lim', v, diago_log_w, v)  # matrix logarithm of sigma --------> tutaj trzeba poprawić i jeszcze się zastanowić
        #log_sigma = np.einsum('lij,lj,lkj->lik',v,np.log(w),v) #matrix logarithm of sigma --------> to prawdopodobnie daje złe wyniki
        #log_dets = -.5 * np.einsum('kii->k',log_sigma)#we use identity log(det(sigma)) = tr(log(sigma))
        log_dets = -0.5 * np.log(np.linalg.det(self.__sigma))

        log_values = -.5 * self.__d * np.log(2 * np.pi) + log_dets[:, np.newaxis] + under_exp_values
        # newaxis przy log_dets sprawia, że dodajemy w poziomie
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

        log_lik = np.ravel(scipy.special.logsumexp(log_liks + self.__log_pi,axis=1)) /float(self.__M)  # multiply by proportions and normalize per training vector
        #zmiana - dodałem "axis=1"
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

        #self.__best_sigma = np.zeros((self.__d, self.__K))
        self.__best_sigma = np.zeros((self.__K,self.__d,self.__d))

        self.__best_kernel_width = np.zeros((self.__K,))


    def __initialize_sigma(self):

        diff = self.__mu[np.newaxis, :, :].transpose((2, 0, 1)) - self.train_set[np.newaxis, :, :].transpose((0, 2, 1))
        #diff ma wymiar (liczba komponentów, liczba próbek treningowych,wymiar wektora średniej)

        sigma = np.einsum('ijl,ijk->ilk',diff,diff)
        #sprawdzone - wygląda spoko

        sigma /= np.float(self.__K - 1)

        sigma = self.__well_condition(sigma)

        return sigma

    def __well_condition(self,sigma):
        w,v = np.linalg.eig(sigma) #w contains eigenvalues, v contains eigenvectors
        w_well = np.real(np.array(np.where(w > 1.e-6, w, 1.e-6)))

        diago_w = np.einsum('ij,li->lji', np.eye(2), np.real(w_well)) #tworzenie macierzy diagonalnych z wektorów wartości własnych
        sigma_well = np.einsum('lij,ljk,lmk->lim', v, diago_w, v)

        return sigma_well

    def __update_log_responsibilities(self):

        self.__log_responsibilities = self.__log_responsibility_mat()

        return self.__log_responsibilities

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

    def __mu_all(self):

        aux = np.exp(self.__log_responsibilities - np.ravel(self.__log_nk)[:, np.newaxis])

        mu = np.einsum('km, dm -> dk', aux, self.train_set)

        return mu

    def __update_sigma(self):

        self.__sigma = self.__sigma_all()

    def __sigma_all(self):

        diff = self.__mu[np.newaxis, :, :].transpose((2, 0, 1)) - self.train_set[np.newaxis, :, :].transpose((0, 2, 1))

        #diff2 = diff * diff
        diff2 = np.einsum('ijl,ijk->ijlk', diff, diff)
        #liczba komponentów x liczba próbek x 2 x 2

        aux = np.exp(self.__log_responsibilities - np.ravel(self.__log_nk)[:, np.newaxis])
        #liczba komponentów x liczba próbek

        sigma = np.einsum('km, kmij -> kij', aux, diff2)
        #liczba komponentów x 2 x 2

        sigma = self.__well_condition(sigma)# for improvement of the numerical stability of the algorithm
        #sigma[sigma < 1e-16] = 1e-16  # for improvement of the numerical stability of the algorithm

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

        self.__update_mu()  # update means ----> dlaczego to jest zakomentowane???

        self.__update_sigma()  # update covariance matrices 

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

        e = -np.sum(np.exp(p) * p)  # entropy -----> entropia rozkładu dyskrentego jest inna niż entropia rozkładu ciągłego
        #if which_component==0:
        #h = self.__entropy(which_component)

        return np.exp(e)
        #return np.exp((e+h)*0.5)  # return effective kernel width in samples (may be fractional)

    def __effective_kernel_width_all(self):

        p = self.__log_liks - np.ravel(scipy.special.logsumexp(self.__log_liks, axis=1))[:, np.newaxis] #sumuje w poziomie --> poziom to próbki, pion to komponenty

        e = -np.sum(np.exp(p) * p, axis=1)
        #h = np.array([self.__entropy(i) for i in range(self.__K)])

        #return np.exp((e+h)*0.5)
        return np.exp(e)

    """
    Computes entropy in nats for a given gaussian mixture component
    """

    def __entropy(self, which_component):
        h = .5 * np.log(np.linalg.det(2 * np.pi * np.e * self.__sigma[which_component, :, :]))

        return h

    """
    Computes the KL divergence in nats according to a formula developed in the paper "Expected Kullback-Leibler 
    Divergence for Multivariate Gaussians" (which is attached to the project)
    """

    def __expected_kullback_leibler_divergence(self, kernel_width):
        kl_val = [1e17, 1e17, 7.61686, 3.57164, 1.93348, 1.20501, 0.878625, 0.694292, 0.523061, 0.402871, 0.380194, 0.326291,
                  0.281733, 0.252601, 0.220284, 0.210597, 0.182916, 0.17494, 0.164098, 0.151005, 0.143, 0.136969,
                  0.137235, 0.12161, 0.119035, 0.112821, 0.108811, 0.10404, 0.0997703, 0.0966179, 0.0957282, 0.0908467,
                  0.0855276, 0.0862314, 0.0794293, 0.0791152, 0.073312, 0.0749637, 0.0719718, 0.0698264, 0.0696133,
                  0.0686299, 0.0655839, 0.0600585, 0.0590129, 0.0585538, 0.0596021, 0.0572314, 0.0540806, 0.0549783,
                  0.0548308, 0.0535493, 0.0519179, 0.0487457, 0.0480348, 0.0480839, 0.0485404, 0.0465229, 0.0466365,
                  0.045143, 0.0443461, 0.0436428, 0.0412165, 0.0423921, 0.0399941, 0.0398727, 0.0400812, 0.0395565,
                  0.0377191, 0.0381104, 0.0378575, 0.037067, 0.0359523, 0.0351945, 0.035141, 0.0347722, 0.0337265,
                  0.0346782, 0.0317904, 0.0335812, 0.0336339, 0.0321593, 0.0314926, 0.0312823, 0.031285, 0.0313188,
                  0.0304131, 0.0297137, 0.0285722, 0.02916, 0.0293262, 0.0287269, 0.0273999, 0.0292182, 0.0274839,
                  0.0271347, 0.0275205, 0.0263476, 0.025523, 0.0265135, 0.0254395, 0.0249436, 0.0264952, 0.0252911,
                  0.0246177, 0.0248858, 0.0244564, 0.023151, 0.0241122, 0.0226078, 0.0229904, 0.0231192, 0.0231572,
                  0.0224701, 0.023572, 0.0222395, 0.0230271, 0.0219005, 0.0222921, 0.0207921, 0.0213797, 0.020523,
                  0.0209888, 0.0206155, 0.0211731, 0.0207549, 0.0208428, 0.0207263, 0.0199628, 0.0203532, 0.0195115,
                  0.0200781, 0.0194743, 0.0194907, 0.0187466, 0.0194719, 0.018367, 0.0187679, 0.0183034, 0.018566,
                  0.0186957, 0.0188008, 0.0179449, 0.0183805, 0.0185144, 0.0176898, 0.0181026, 0.0169336, 0.0167282,
                  0.0167446, 0.0168561, 0.01687, 0.016804, 0.0166817, 0.0163657, 0.0161179, 0.0168385, 0.015787,
                  0.0158237, 0.0160259, 0.0159201, 0.016165, 0.0160591, 0.0160028, 0.0156289, 0.0151054, 0.0157096,
                  0.0158995, 0.0152133, 0.014494, 0.0156152, 0.0144547, 0.0145592, 0.0145714, 0.0146889, 0.014712,
                  0.0148254, 0.0145735, 0.0137793, 0.0140497, 0.0149892, 0.0143504, 0.01404, 0.0139598, 0.013928,
                  0.0140183, 0.013439, 0.0135029, 0.0134779, 0.0133723, 0.0135401, 0.0136206, 0.0131313, 0.0136496,
                  0.0132761, 0.0131046, 0.0126652, 0.0127704, 0.0127823, 0.0126326, 0.0127244, 0.0125424, 0.0126286,
                  0.0122185, 0.0122834, 0.0127471, 0.0124437, 0.0122588, 0.0118352, 0.01227, 0.0119693, 0.012127,
                  0.0119773, 0.0121275, 0.0118019, 0.0122205, 0.0117433,
                  0.0115687, 0.0117035, 0.0118512, 0.0111757, 0.0114456, 0.0117721, 0.0115585, 0.011252, 0.0112884,
                  0.0110039, 0.0114985, 0.0110636, 0.0112961, 0.0109929, 0.0112702, 0.0108753, 0.010892, 0.011393,
                  0.0108204, 0.0107544, 0.0105334, 0.0105453, 0.00995563, 0.010636, 0.0105431, 0.0105775, 0.0106496,
                  0.0101615, 0.0101751, 0.0103164, 0.010427, 0.0105463, 0.0101339, 0.010074, 0.0102841, 0.0101619,
                  0.009691, 0.00990087, 0.00973575, 0.00971856, 0.00976254, 0.00988482, 0.00990628, 0.00965732,
                  0.00942487, 0.00962248, 0.00965497, 0.00954072, 0.00970285, 0.00967235, 0.00956675, 0.00934804,
                  0.00940825, 0.00882514, 0.00891107, 0.00936847, 0.00923333, 0.00915541, 0.00925006, 0.00925772,
                  0.00892569, 0.00914414, 0.00894971, 0.00943672, 0.00891993, 0.00903876, 0.00882699, 0.00891053,
                  0.00890751, 0.00873418, 0.00891757, 0.00839776, 0.00858988, 0.00875312, 0.00854132, 0.00896412,
                  0.00861313, 0.00856427, 0.00860146, 0.00864168, 0.00851362, 0.00852567, 0.00836996, 0.00848384,
                  0.00824523, 0.00843788, 0.00852701, 0.00850288, 0.00816566, 0.00833788, 0.00822104, 0.0083006,
                  0.00822798, 0.00781979, 0.00809214, 0.00796783, 0.0079134, 0.00834575, 0.00817318, 0.00791531,
                  0.00795342, 0.00818357, 0.00798441, 0.00794028, 0.00779576, 0.00809724, 0.00793879, 0.00809122,
                  0.00797924, 0.00788901, 0.00796216, 0.00797467, 0.00758163, 0.00750884, 0.00766579, 0.0075524,
                  0.00756504, 0.00750947, 0.007557, 0.00769701, 0.00725741, 0.00758928, 0.00739336, 0.00755581,
                  0.00732737, 0.0075401, 0.00719396, 0.00733421, 0.00718423, 0.00742689, 0.00723333, 0.00721869,
                  0.0073088, 0.00708677, 0.00723663, 0.00720881, 0.00716277, 0.00730259, 0.00713608, 0.00709807,
                  0.00717526, 0.00677888, 0.00706042, 0.00690356, 0.0070374, 0.00688405, 0.00693215, 0.00711014,
                  0.00699985, 0.00722337, 0.00723113, 0.00703235, 0.00693357, 0.00691247, 0.00667866, 0.00663952,
                  0.00669672, 0.00663483, 0.00650182, 0.00635884, 0.00673547, 0.00654602, 0.00660859, 0.00656313,
                  0.00673361, 0.00666121, 0.00654513, 0.00652524, 0.00650237, 0.00635136, 0.00655595, 0.00626191,
                  0.00646381, 0.00632988, 0.00659053, 0.00635403, 0.00658221, 0.00653649, 0.00642961, 0.00627738,
                  0.00644303, 0.00615859, 0.00615425, 0.00655524, 0.00657711, 0.00622143, 0.00620663, 0.0062362,
                  0.00614497, 0.00613437, 0.00616269, 0.00602823, 0.00630963, 0.00604973, 0.00589715, 0.00608343,
                  0.00599622, 0.00608011, 0.00613202, 0.00598452, 0.00609512, 0.00610681, 0.00596073, 0.00602234,
                  0.00613123, 0.00569371, 0.0057111, 0.00591205, 0.00582189, 0.00590858, 0.00575628, 0.00578274,
                  0.00610638, 0.00595204, 0.00582097, 0.00577386, 0.00584919, 0.00572665, 0.00577898, 0.00582897,
                  0.00577196, 0.00572061, 0.00551691, 0.0056829, 0.00564545, 0.00572495, 0.00577977, 0.00575814,
                  0.00567872, 0.00554558, 0.00547831, 0.00551935, 0.00559673, 0.00563887, 0.00554781, 0.00562392,
                  0.00535375, 0.00541017, 0.00527785, 0.00541559, 0.00537132, 0.00562858, 0.0052717, 0.00557141,
                  0.00556756, 0.00543454, 0.00534027, 0.00555792, 0.00566045, 0.00551107, 0.00541254, 0.00533293,
                  0.00526229, 0.00549583, 0.00518183, 0.00542095, 0.00513004, 0.00528862, 0.00519328, 0.00539081,
                  0.00524956, 0.00525988, 0.0052618, 0.0054697, 0.0051614, 0.00535221, 0.00520074, 0.00516413,
                  0.00528561, 0.00510492, 0.00527851, 0.00498773, 0.00502874, 0.00500081, 0.00528656, 0.00518175,
                  0.00513762, 0.00517254, 0.00479206, 0.00521134, 0.00493966, 0.00537282, 0.00507066, 0.00504638,
                  0.00505583, 0.0050395, 0.00483178, 0.00496704, 0.00494659, 0.00488834, 0.00490909, 0.00498534,
                  0.00497239, 0.00481174, 0.00485047, 0.00485148, 0.00491011, 0.00493987, 0.00479238, 0.00499793,
                  0.00500554, 0.00479796, 0.00499678, 0.00478871, 0.00487864, 0.00476745, 0.00491326, 0.00465619,
                  0.00460527, 0.00463156, 0.0048174, 0.00483943, 0.00466716, 0.00492114, 0.00465327, 0.00459663,
                  0.00469553, 0.00471228, 0.00466211, 0.00459558, 0.00455968, 0.00476581, 0.00458659, 0.00463393,
                  0.00478801, 0.00459005, 0.00464781, 0.00465549, 0.00456797, 0.00464373, 0.00458642, 0.00460619,
                  0.00436345, 0.00473141, 0.00458545, 0.00454851, 0.00449678, 0.00460828, 0.00444348, 0.00446808,
                  0.00457813, 0.00469321, 0.00451276, 0.00452145, 0.00465913, 0.0044821, 0.00433084, 0.00438941,
                  0.00440004, 0.00436715, 0.00439857, 0.00446234, 0.00444093, 0.00458575, 0.00444103, 0.00447651,
                  0.00445598, 0.00442709, 0.00457098, 0.00433766, 0.00445886, 0.00443317, 0.00437372, 0.00442099,
                  0.00444746, 0.00427274, 0.00437764, 0.00430287, 0.00432989, 0.00416624, 0.00431959, 0.00426457,
                  0.00410719, 0.00421755, 0.00438084, 0.00421095, 0.00418929, 0.0041873, 0.00432957, 0.00430412,
                  0.00432871, 0.00413341, 0.00423968, 0.00420512, 0.00425464, 0.00425272, 0.00413582, 0.00423113,
                  0.00406893,
                  0.00415664, 0.00412395, 0.00409132, 0.00417249, 0.0040867, 0.00408759, 0.00408141, 0.00403065,
                  0.00396894, 0.00410093, 0.00409209, 0.00393229, 0.00401394, 0.00409174, 0.00415652, 0.00407167,
                  0.0040121, 0.0041372, 0.00391883, 0.00393847, 0.00405196, 0.00392987, 0.00397125, 0.00403948,
                  0.00394173, 0.00378202, 0.00391362, 0.0040457, 0.00397905, 0.0040333, 0.00389108, 0.00399019,
                  0.00395193, 0.00393094, 0.00398435, 0.00402269, 0.00392089, 0.00391103, 0.00378529, 0.0038481,
                  0.00392743, 0.00383219, 0.00378551, 0.00387275, 0.00384703, 0.00381509, 0.00380013, 0.00381845,
                  0.00378434, 0.00389636, 0.00382464, 0.00387412, 0.00388549, 0.00386315, 0.00373133, 0.00372041,
                  0.00393777, 0.00379339, 0.00371071, 0.00367224, 0.00381165, 0.00382719, 0.00366045, 0.00388254,
                  0.00372675, 0.00379459, 0.00363793, 0.00377187, 0.003765, 0.00376832, 0.0036573, 0.00364045,
                  0.00372673, 0.00361792, 0.00369979, 0.00360791, 0.00366289, 0.00368269, 0.00363065, 0.00370661,
                  0.00363502, 0.00367659, 0.00365421, 0.00372528, 0.003576, 0.00364858, 0.00382958, 0.00371724,
                  0.00368848, 0.00368947, 0.00365316, 0.00367844, 0.00358445, 0.00359008, 0.0036077, 0.00363544,
                  0.0035856, 0.00357971, 0.0036228, 0.00361563, 0.00351398, 0.0035517, 0.00356755, 0.00355577,
                  0.00348919, 0.00352475, 0.00345253, 0.00349033, 0.00347809, 0.00347846, 0.00343582, 0.00348286,
                  0.00351044, 0.00355002, 0.00358516, 0.003493, 0.00345716, 0.00343487, 0.00353088, 0.0035496,
                  0.00333503, 0.00346332, 0.0035748, 0.00351211, 0.00356028, 0.00337385, 0.0033889, 0.00343964,
                  0.00353515, 0.00329274, 0.0034871, 0.0033505, 0.00351224, 0.00344156, 0.00327419, 0.00332056,
                  0.00345573, 0.00335628, 0.00343202, 0.00339274, 0.00337735, 0.00330229, 0.0035506, 0.00335428,
                  0.00334349, 0.00320666, 0.00340006, 0.00333641, 0.00332736, 0.00330358, 0.00337659, 0.00324908,
                  0.00334568, 0.00321832, 0.00335456, 0.00332467, 0.00336768, 0.00326885, 0.00333332, 0.00340212,
                  0.00339688, 0.00332003, 0.00333064, 0.00322135, 0.00324928, 0.00329089, 0.00338995, 0.00331032,
                  0.00331803, 0.00317147, 0.00320814, 0.00314295, 0.00323692, 0.00327746, 0.00320891, 0.003275,
                  0.0032422, 0.00313691, 0.00325782, 0.00317023, 0.00325735, 0.00324195, 0.00310544, 0.00313071,
                  0.00320209, 0.00319958, 0.00321835, 0.00324655, 0.00313584, 0.00319688, 0.00299539, 0.00317871,
                  0.00317572, 0.00323449,
                  0.00308598, 0.00305455, 0.00314988, 0.00308567, 0.0030363, 0.00304506, 0.00307033, 0.00305563,
                  0.00305785, 0.00315358, 0.00308774, 0.0031187, 0.00315055, 0.00298922, 0.00303752, 0.00313292,
                  0.00297228, 0.00309156, 0.00314049, 0.00306281, 0.00303717, 0.00311324, 0.00308902, 0.0029828,
                  0.00292162, 0.00297825, 0.00301082, 0.00313022, 0.00308328, 0.00301186, 0.00304873, 0.0029728,
                  0.00296453, 0.00304051, 0.00300843, 0.00301861, 0.00300432, 0.00300091, 0.00298984, 0.00292813,
                  0.00291852, 0.0029981, 0.0028936, 0.00301434, 0.00299089, 0.00297004, 0.00305243, 0.00304158,
                  0.00299615, 0.0029772, 0.0029499, 0.00310332, 0.00306951, 0.00300631, 0.0029722, 0.00291826,
                  0.00292876, 0.00290656, 0.00299347, 0.0028473, 0.00282879, 0.00290382, 0.00284756, 0.00296723,
                  0.00284894, 0.0029426, 0.00284414, 0.00286532, 0.00289208, 0.00287849, 0.00289685, 0.00285509,
                  0.0027864, 0.00286676, 0.00291372, 0.00274864, 0.00288117, 0.00282988, 0.00285673, 0.00286997,
                  0.00280549, 0.00289745, 0.0028125, 0.00278749, 0.00282862, 0.00287552, 0.00285646, 0.00288625,
                  0.0027397, 0.00279703, 0.00278547, 0.00292725, 0.00289112, 0.00276878, 0.00282393, 0.00270398,
                  0.00278771, 0.00274646, 0.00271242, 0.00281754, 0.00269562, 0.00288015, 0.0027444, 0.00276236,
                  0.00271574, 0.00281065, 0.00274178, 0.00276013, 0.00278178, 0.00271836, 0.00274933, 0.00281329,
                  0.0026916, 0.00264294, 0.0027305, 0.00260524, 0.0027036, 0.00278114, 0.0027616, 0.00276963,
                  0.00275752, 0.00270724, 0.00264202, 0.00271448, 0.00270573, 0.0026486, 0.00265882, 0.0027197,
                  0.0027707, 0.00268009, 0.00270978, 0.00268322, 0.00268362, 0.00270217, 0.002618, 0.00267758,
                  0.00262214, 0.00271195, 0.00264954, 0.0026509, 0.00274516, 0.00267409, 0.00261628, 0.00265943,
                  0.00271045, 0.00275206, 0.00273225, 0.00258994, 0.00260194, 0.00269026, 0.00267267, 0.00266282,
                  0.00270984, 0.0026072, 0.00257548, 0.00272528, 0.00264234, 0.00268572, 0.00266409, 0.00258058,
                  0.00263342, 0.0026024, 0.00262339, 0.00261377, 0.00256808, 0.00265955, 0.00258559, 0.0025835,
                  0.00257439, 0.00267765, 0.00254819, 0.0025135, 0.0025898, 0.00258469, 0.00254139, 0.00256115,
                  0.00270299, 0.00253053, 0.00263439, 0.00260213, 0.00248472, 0.00262936, 0.00253552, 0.00253077,
                  0.0025529, 0.00258095, 0.00256452, 0.00251718, 0.00245697, 0.00253716, 0.00241109, 0.00247849,
                  0.00253198, 0.00248975, 0.00252391,
                  0.00250707, 0.00256004, 0.00245168, 0.00252693, 0.00247699, 0.0025156, 0.00249221, 0.00255033,
                  0.00247528, 0.00247401, 0.00240561, 0.00252229, 0.0025675, 0.00245601, 0.00254427, 0.00250625,
                  0.00252132, 0.00250688, 0.00237315, 0.0025623, 0.00248281, 0.00246195, 0.00237443, 0.00240334,
                  0.00243842, 0.00249521, 0.00245931, 0.0024534, 0.00253756, 0.00244317, 0.00244703, 0.00244622,
                  0.00249103, 0.00241245, 0.00252468, 0.00259266, 0.00247522, 0.002512, 0.00239609, 0.00241705,
                  0.00236024, 0.00238739, 0.00243189, 0.00247513, 0.00241621, 0.0024856, 0.00235338, 0.00240958,
                  0.00240791, 0.00238595, 0.00234098, 0.00236778, 0.00244847, 0.00230043, 0.00235821, 0.00234401,
                  0.00236874, 0.00236198, 0.00239788, 0.00244162, 0.00245329, 0.00230645, 0.00236645, 0.00242145,
                  0.00241766]
        if kernel_width < len(kl_val):
            ekl = np.interp(kernel_width-1,np.array(range(len(kl_val))),np.array(kl_val))
        else:
            ekl = kl_val[-1]
        return ekl

    def fit(self, number_of_iterations=300):

        self.__initialize()

        log_lik = self.__log_likelihood_per_training_vector()

        i = 0

        print(f"Log likelihood per training vector at iteration {i} is equal {log_lik[0]}")

        scores = [[] for i in range(self.__K)]

        #
        #min_x, max_x, min_y, max_y = self.bounds_tmp()
        #

        for i in range(1, number_of_iterations):
            self.__e_step()

            self.__m_step()

            log_lik = self.__log_likelihood_per_training_vector()

            #rysowanie pomocnicze
            # plt.figure(6)
            # plt.clf()
            # plt.ylim(min_y,max_y)
            # plt.xlim(min_x,max_x)
            # plt.scatter(self.__mu[0],self.__mu[1],c='green',s=1)
            # plt.show(block=False)
            # plt.pause(0.1)
            #
            # plt.figure(7)
            # plt.clf()
            # plt.plot(self.__pi)
            # plt.show(block=False)
            #koniec rysowania pomocniczego

            kernel_width: np.array = self.__effective_kernel_width_all()

            for k in range(self.__K):

                s = self.__score(k)

                scores[k].append(s)

                if s < self.__best_scores[k]:
                    self.__best_scores[k] = s

                    self.__best_mu[:, k] = copy.copy(self.__mu[:, k])

                    #self.__best_sigma[:,k] = copy.copy(self.__sigma[:,k])
                    self.__best_sigma[k,:,:] = copy.copy(self.__sigma[k,:,:])

                    self.__best_kernel_width[k] = kernel_width[k]

            print(f"Log likelihood per training vector at iteration {i} is equal {log_lik[0]}")

            print(f"Entropy of the 1st component {self.__entropy(1)}")
            #
            print(f"Effective kernel width {self.__effective_kernel_width(1)}")
            #
            print(f"Score per 1st component {self.__score(1)}")
            #
            print(f"KL distance {self.__expected_kullback_leibler_divergence(self.__effective_kernel_width(1))}")

            #scores.append(self.__score(1))

        self.__mu = self.__best_mu
        self.__sigma = self.__best_sigma

        plt.figure(9)
        plt.plot(scores[0])
        plt.plot(scores[1])
        plt.plot(scores[2])
        plt.show()

        self.__initialize_pi()
        for i in range(4):
            log_lik = self.__log_likelihood_per_training_vector()

            self.__update_log_responsibilities()

            self.__update_log_nk()

            self.__update_log_n()

            self.__update_pi()
            print(f"Final log_lik {log_lik}")

    def get_gmm(self):
        gmm = {'mu': self.__best_mu, 'sigma': self.__best_sigma, 'sigma_inv': np.linalg.inv(self.__best_sigma),
               'log_pi': self.__log_pi}
        return gmm

    def bounds_tmp(self):
        min_ = np.min(self.train_set, axis=1)
        max_ = np.max(self.train_set, axis=1)
        min_x = min_[0]
        min_y = min_[1]
        max_x = max_[0]
        max_y = max_[1]
        aux = max_ - min_
        width = aux[0]
        height = aux[1]

        max_x += .5 * width
        max_y += .5 * height
        min_x -= .5 * width
        min_y -= .5 * height

        return min_x, max_x, min_y, max_y