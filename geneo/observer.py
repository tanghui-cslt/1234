import numpy as np
import platform
if platform.system() == "linux":
    import matplotlib
    matplotlib.use("Agg")
import dionysus as di
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from tqdm import tqdm
from itertools import combinations
import parmap
from geneo.read_data import DataSet
from geneo.gaussian_ieneo import IENEO 
from geneo.constants import PERSISTENCE_PARAMS
from geneo.utils import bin_coeff,plot_diagram, is_included, get_order_of_magnitude


class Observer:
    """An observer can be described as an entity capable of measuring similarity
    between signals because equipped with a set of measuring functions.

    Parameters
    ----------
    geneos : list
        list of operators
    observed_signal_dimension : int
        Dimension of the observed signal. It is used to generate the operators
    persistence_params : dict
        Dictionary of parameters for the computation of persistet homology. See
        :const:`geneo.constants.PERSISTENCE_PARAMS`

    Attributes
    ----------
    operators : list
        List of operators available to the observer to measure the signal
    """
    def __init__(self, geneos, observed_signal_dimension=2,
                 persistence_params=None, convolve_params=None):
        ### NOTE: probably an input  "operators parameters" will also be needed
        self.operators = geneos
        self._num_of_operators = len(geneos)
        self.observable_dim = observed_signal_dimension
        self.persistence_params = persistence_params
        self.convolve_params = convolve_params
        [f.generate(dim = self.observable_dim)
         for f in self.operators]
        self.sampled_operators = None
        self.selected_operators = None


    @property
    def persistence_params(self):
        return self._persistence_params


    @persistence_params.setter
    def persistence_params(self, new_params):
        if new_params is None:
            new_params = PERSISTENCE_PARAMS
        self._persistence_params = new_params


    @property
    def hom_deg(self):
        return self._hom_deg


    @hom_deg.setter
    def hom_deg(self, new_hom_deg):
        if not hasattr(self, "hom_deg"):
            self._hom_deg = new_hom_deg
        else:
            if new_hom_deg != self.hom_deg:
                message = ("The homoloy degree used for comparisons was" +
                "{} and cannot be modified to {}".format(self.hom_deg, new_hom_deg))
                raise ValueError(message)


    @property
    def num_of_operators(self):
        return self._num_of_operators


    @staticmethod
    def evaluate_operator_on_signal_class(operator, signals, hom_deg,
                                          persistence_params, convolve_params):
        #convolve operator with all signals and compute hom_degth persistence
        #diagram

        diags = [operator.convolve_and_get_persistence(signal,
                               persistence_params =persistence_params,
                               convolve_params =convolve_params)[hom_deg]
                 for signal in signals]
        #get all the possible pairs of persistence diagrams(without repetitions)
        # hence pairs of the form
        # ( D_{hom_deg}(operator(signal_i)), D_{hom_deg}(operator(signal_j)) )
        diags_pairs = combinations(diags, 2)
        # compute the pairwise bottleneck distance
        distances = [di.bottleneck_distance(*diags) for diags in diags_pairs]
        # return the max
        #print("len distances ", len(distances), "len(signals) ",len(signals))
        maximum = np.max(distances)
        return maximum


    def select_per_class(self, signal_class, hom_deg,
                                              threshold):
        #evaluate in parallel all the operators on the signal samples belonging
        #to a fixed class
        evals = parmap.map(self.evaluate_operator_on_signal_class,
                           self.operators, signal_class, hom_deg,
                           self.persistence_params, self.convolve_params,
                           pm_pbar=True)
        #print("num self.operators ",len(self.operators))
        #compute the standard deviation of the evaluations
        std = np.std(evals)
        #select the indices of the operators that are below std * threshold
        return [i for i, e in enumerate(evals) if e < threshold * std]


    def select_filters_on_signal_set(self, train_signals, hom_deg = 1,
                                     threshold = 1.5, max_ops_per_class = None):
        """
        We select operators that represent in a similar way a set of signals
        belonging to the same class. The algorithm is structured as follows

        1. we associate to each operator F
        sel_{F,c} = max d_b( D_{hom_def}(F(\phi_i)), D_{hom_def}(F(\phi_j)) )
        for every pair (\phi_i, \phi_j) of signals belonging to the same class.

        2. We then consider the set
        Sel(c) = {sel_{F_i,c} for every F_i \in\mathcal{F}}
        where $\mathcal{F}$ is the set of all operators associated with the
        observer.

        3. We consider acceptable those operators F* such that
        sel_{F*,c} < `threshold` * std(Sel(c))
        """
        self._hom_deg = hom_deg
        indices = []

        for c in tqdm(train_signals, desc = "Selecting operators"):
            signals = train_signals[c]
            indices_c = self.select_per_class(signals, hom_deg, threshold)
            if max_ops_per_class is not None:
                indices_c = indices_c[:max_ops_per_class]
            indices.append(indices_c)
        # print("indices ",indices)
        indices = np.unique(np.concatenate(indices)).astype("int")
        self.selected_operators = [self.operators[ind]
                                   for ind in indices]

    @staticmethod
    def compute_ops_dist(op_pair, signals, persistence_params, convolve_params,
                         hom_deg, func):
        pair_eval = [op_pair[0].get_distance_on_signal(op_pair[1], s, hom_deg,
                                                       persistence_params,
                                                       convolve_params)
                     for s in signals]
        return func(pair_eval)


    def get_pairwise_ops_distance(self, signals, operators=None, func=None):
        if operators is None:
            operators = self.get_operators()
        ops_pairs = list(combinations(operators, 2))
        evals = parmap.map(self.compute_ops_dist, ops_pairs, signals,
                           self.persistence_params, self.convolve_params,
                           self.hom_deg, func, pm_pbar=True)
        return evals


    def get_pairwise_ops_distance_per_class(self, train_signals, indices, func):
        num_classes = len(list(train_signals.keys()))
        distances_per_class = np.zeros((num_classes, len(indices)))

        for i, c in enumerate(tqdm(train_signals, desc = "Sampling operators")):
            signals = train_signals[c]
            # compute pairwise distances of operators on the signal belonging to
            # the same class
            distances_per_class[i,] = self.get_pairwise_ops_distance(signals,
                                                                     func=func)

        return distances_per_class


    @staticmethod
    def find_value_in_cols(matrix, value):
        return np.where(matrix == value)[1]


    def assign_interclass_sampling_score(self, distances, indices):
        distances_argsort = np.argsort(distances, axis=1)
        return [sum(self.find_value_in_cols(distances_argsort, i))
                for i in range(len(indices))]


    def sample_operators(self, train_signals, func=np.max, th_per = 75):
        """
        We sample operators to avoid storing filters that would focus on the
        same or similar characteristic across classes. We proceed as follows:

        1. We compute the pairwise distance between operators F_i and F_j on
           the signals \Phi_c associated to the class c as
           d_c(F_i, F_j) = func( d_B( D(F_1(\phi)), D(F_2(\phi)) ) )

           Remark: pairs of operators are built without without repetitions and
           such that i \neq j for every pair (op_i, op_j)

        2. We organize each vector {d_c(F_i, F_j)} in a matrix called
           `distances`, where each rows is associated to a class c

        3. We associate to each pair of operators (F_i, F_j) an interclass
           contrastive score by considering the sum of the row indices of
           (F_i, F_j) in the cols-wise argsort of the matrix `distances`

           In other words if on n possible pairs of operators the pair (i,j)
           is the most discriminative in a three classes problem, it will have
           score 3*n

        4. Finally we consider those operators that are part of pairs that score
           over the `th_per` percentile of all contrastive scores.

        """
        operators = self.get_operators()
        indices = list(combinations(range(self.num_of_operators), 2))
        # distances are a matrix where each column represents a class and each
        # row is a pair of operators organized according to indices
        distances = self.get_pairwise_ops_distance_per_class(train_signals,
                                                             indices, func)
        #scores determine which operators are further apart with respect to all
        scores = self.assign_interclass_sampling_score(distances, indices)
        #threshold operators
        th = np.percentile(scores, th_per)
        contrastive_pairs_indices = [set(indices[i])
                                     for i, s in enumerate(scores) if s > th]
        contrastive_pairs_indices = set().union(*contrastive_pairs_indices)
        self.sampled_operators = [operators[ind] for ind in
                                  contrastive_pairs_indices]


    def get_distance_signals(self, s1, s2, func = np.max):
        operators = self.get_operators()
        ### TODO parallelize this if it is too slow
        # distances = parmap.map(op.get_signals_distance, operators, self.hom_deg,
        #                        self.persistence_params, self.convolve_params,
        #                        pm_pbar=True)
        distances = [op.get_signals_distance(s1, s2, self.hom_deg,
                                             self.persistence_params,
                                             self.convolve_params)
                     for op in operators]
        return func(distances)


    def get_operators(self):
        #XXX It will become a property once the code is stable
        if self.sampled_operators is not None:
            ops = self.sampled_operators
        elif self.selected_operators is not None:
            ops = self.selected_operators
        else:
            ops = self.operators
        self._num_of_operators = len(ops)
        return ops

    def get_operators_params(self):
        return [f.get_params() for f in self.get_operators()]

    def save(self, path):
        np.save(path, self)

    @classmethod
    def load(cls, path):
        #A = np.load(path,allow_pickle=True).item()
        #print("\n================\n")
        #print(A)
        return np.load(path,allow_pickle=True).item()


    def visualise_conv(self, image, convolve_params = None,
            persistence_params = None, axes = None, hom_deg = 1):
          """Visualise the 1 and 2-dimensional rendering of the kernel, an image,
          the result of the convolution and the corresponding persistent diagram
          of degree `hom_deg`
          """
          if axes is None:
              fig, axs = plt.subplots(2,3)
              axs = axs.ravel()
          if persistence_params is None:
              persistence_params = PERSISTENCE_PARAMS
              convolved_image = self.convolve(image, params = convolve_params)
              dgms = self.convolve_and_get_persistence(convolved_image,
                                                      convolve = False,
                                                      persistence_params =
                                                      persistence_params)
              axs.ravel()
              axs[0].plot(self.g)
              
              axs[1].imshow(self.kernel)
              axs[2].imshow(image)
              axs[3].imshow(convolved_image)

              plot_diagram(dgms[hom_deg], ax = axs[4])

if __name__ == "__main__":
    from geneo.utils import init_operators
    ops = init_operators(IENEO)
    observer = Observer(ops)
    data = DataSet('mnist', num_samples_from_training = 10)
    #visualise convolution with a specific image
    print(ops)
    ops[0].visualise_conv(data.x_train[0])
    #observer.visualise_conv(data.x_train[0])
