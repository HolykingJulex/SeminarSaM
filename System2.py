from typing import Any

import numpy

from numpy import ndarray


import json
import random

import numpy



def thermal_field(
    temperature: ndarray[(Any, 3), float],
    magnitude_spin_moment: ndarray[Any, float],
    damping: float,
    deltat: float,
    gyromagnetic: float,
    kB: float,
) -> ndarray[(Any, 3), float]:
    N = len(magnitude_spin_moment)
    gamma = numpy.random.normal(size=(N, 3))
    values = (2 * damping * kB * temperature) / (
        gyromagnetic * magnitude_spin_moment * deltat
    )
    values = numpy.sqrt(values)
    values = numpy.repeat(values, 3).reshape((len(values), 3))
    return gamma * values



def magnetic_field(
    magnetic_fields: ndarray[(Any, 3), float]
) -> ndarray[(Any, 3), float]:
    return magnetic_fields


def compute_exchange_energy(
    state: ndarray[(Any, 3), float],
    exchanges: ndarray[(Any, Any), float],
    neighbors: ndarray[(Any, Any), float],
) -> float:
    total = 0
    N = len(state)
    for i in range(N):
        state_i = state[i]
        exchanges_i = exchanges[i]
        neighbors_i = state[neighbors[i]]
        total -= (exchanges_i * (state_i * neighbors_i).sum(axis=1)).sum()
    return 0.5 * total



def compute_anisotropy_energy(
    state: ndarray[(Any, 3), float],
    anisotropy_constants: ndarray[Any, float],
    anisotropy_vectors: ndarray[(Any, 3), float],
) -> float:
    return -(anisotropy_constants * (state * anisotropy_vectors).sum(axis=1) ** 2).sum()



def compute_magnetic_energy(
    state: ndarray[(Any, 3), float],
    magnitude_spin_moment: ndarray[Any, float],
    magnetic_fields: ndarray[(Any, 3), float],
) -> float:
    return -(magnitude_spin_moment * (state * magnetic_fields).sum(axis=1)).sum()



def dS_llg(
    state: ndarray[(Any, 3), float],
    Heff: ndarray[(Any, 3), float],
    damping: float,
    gyromagnetic: float,
) -> ndarray[(Any, 3), float]:
    alpha = -gyromagnetic / (1 + damping * damping)
    cross1 = numpy.cross(state, Heff)
    cross2 = numpy.cross(state, cross1)
    return alpha * (cross1 + damping * cross2)



def normalize(matrix: ndarray[(Any, 3), float]) -> ndarray[(Any, 3), float]:
    norms = numpy.sqrt((matrix * matrix).sum(axis=1))
    norms_repeated = numpy.repeat(norms, 3)
    norms_repeated_reshaped = norms_repeated.reshape((len(norms), 3))
    return matrix / norms_repeated_reshaped


def integrate(
    state: ndarray[(Any, 3), float],
    magnitude_spin_moment: ndarray[Any, float],
    temperature: ndarray[Any, float],
    damping: float,
    deltat: float,
    gyromagnetic: float,
    kB: float,
    magnetic_fields: ndarray[(Any, 3), float],
    exchanges: ndarray[(Any, Any), float],
    neighbors: ndarray[(Any, Any), float],
    anisotropy_constants: ndarray[Any, float],
    anisotropy_vectors: ndarray[(Any, 3), float],
) -> float:
    # compute external fields. These fields does not change
    # because they don't depend on the state
    Hext = thermal_field(
        temperature, magnitude_spin_moment, damping, deltat, gyromagnetic, kB
    )
    Hext += magnetic_field(magnetic_fields)

    # predictor step

    # compute the effective field as the sum of external fields and
    # spin fields
    Heff = Hext + exchange_interaction_field(
        state, magnitude_spin_moment, exchanges, neighbors
    )
    Heff = Heff + anisotropy_interaction_field(
        state, magnitude_spin_moment, anisotropy_constants, anisotropy_vectors
    )

    # compute dS based on the LLG equation
    dS = dS_llg(state, Heff, damping, gyromagnetic)

    # compute the state_prime
    state_prime = state + deltat * dS

    # normalize state_prime
    state_prime = normalize(state_prime)

    # corrector step

    # compute the effective field prime by using the state_prime. We
    # use the Heff variable for this in order to reutilize the memory.
    Heff = Hext + exchange_interaction_field(
        state_prime, magnitude_spin_moment, exchanges, neighbors
    )
    Heff = Heff + anisotropy_interaction_field(
        state_prime, magnitude_spin_moment, anisotropy_constants, anisotropy_vectors
    )

    # compute dS_prime employing the Heff prime and the state_prime
    dS_prime = dS_llg(state_prime, Heff, damping, gyromagnetic)

    # compute the new state
    integrate = state + 0.5 * (dS + dS_prime) * deltat

    # normalize the new state
    return normalize(integrate)



def exchange_interaction_field(
    state: ndarray[(Any, 3), float],
    magnitude_spin_moment: ndarray[Any, float],
    exchanges: ndarray[(Any, Any), float],
    neighbors: ndarray[(Any, Any), float],
) -> ndarray[(Any, 3), float]:
    N = len(magnitude_spin_moment)
    out = numpy.zeros(shape=(N, 3))
    for i in range(N):
        exchanges_i = exchanges[i]
        exchanges_i_repeated = numpy.repeat(exchanges_i, 3)
        exchanges_i_repeated_reshaped = exchanges_i_repeated.reshape(
            (len(exchanges_i), 3)
        )
        neighbors_i = state[neighbors[i]]
        out[i] = (exchanges_i_repeated_reshaped * neighbors_i).sum(
            axis=0
        ) / magnitude_spin_moment[i]
    return out



def anisotropy_interaction_field(
    state: ndarray[(Any, 3), float],
    magnitude_spin_moment: ndarray[Any, float],
    anisotropy_constants: ndarray[Any, float],
    anisotropy_vectors: ndarray[(Any, 3), float],
) -> ndarray[(Any, 3), float]:
    values = (
        anisotropy_constants
        * (state * anisotropy_vectors).sum(axis=1)
        / magnitude_spin_moment
    )
    values_repeated = numpy.repeat(values, 3)
    values_repeated_reshaped = values_repeated.reshape((len(values), 3))
    return 2 * values_repeated_reshaped * anisotropy_vectors


from collections.abc import Iterable
from numbers import Real



class Bucket:
    """This is a class to match the sizes of two attributes. For this case, the
    attributes are the ``temperature`` and the ``field``.

    :param bucket_1: It gets the temperature information of the ``simulation_file``.
    It receives one of the two attributes, temperature or field.
    :type bucket_1: float/list/dict
    :param bucket_2: It gets the field information of the ``simulation_file``. It
    receives one of the two attributes, temperature or field.
    :type bucket_2: float/list/dict
    """

    def __init__(self, structure):
        """The constructor for Bucket class.

        :param structure: It receives the two attributes (one at a time), temperature
        or field. It is responsible for determining the ``type`` of attribute .
        :type structure: float/list/dict
        """
        if isinstance(structure, dict):
            start = structure["start"]
            final = structure["final"]
            step = structure["step"]
            step = numpy.sign(final - start) * abs(step)
            self.values = numpy.arange(start, final + step, step)
        elif isinstance(structure, Iterable):
            self.values = structure
        elif isinstance(structure, Real):
            self.values = [structure]
        else:
            raise Exception("[Bucket for temperature and field] No supported format.")

    def __len__(self):
        """It is a function to determine the lenght of the two attributes."""
        return len(self.values)

    def __iter__(self):
        """It is a function that create an object which can be iterated one element at
        a time.
        """
        return iter(self.values)

    @staticmethod
    def match_sizes(bucket_1, bucket_2):
        """It is a function decorator, it is an instance for read the attributes and
        match it sizes.

        :param bucket_1: It gets the temperature information of the ``simulation_file``.
        :type bucket_1: float/list
        :param bucket_2: It gets the field information of the ``simulation_file``.
        :type bucket_2: float/list

        :return: An object that has the same size of bucket_2.
        :rtype: Object
        :return: An object that has the same size of bucket_1.
        :rtype: Object
        """
        if len(bucket_1) == len(bucket_2):
            return bucket_1, bucket_2

        if len(bucket_1) < len(bucket_2):
            while len(bucket_1) < len(bucket_2):
                bucket_1 = Bucket(bucket_1.values * 2)
            return Bucket(bucket_1.values[: len(bucket_2)]), bucket_2

        if len(bucket_2) < len(bucket_1):
            while len(bucket_2) < len(bucket_1):
                bucket_2 = Bucket(bucket_2.values * 2)
            return bucket_1, Bucket(bucket_2.values[: len(bucket_1)])







def get_random_state(num_sites):
    random_state = numpy.random.normal(size=(num_sites, 3))
    norms = numpy.linalg.norm(random_state, axis=1)
    random_state = [vec / norms[i] for i, vec in enumerate(random_state)]
    return random_state


class Simulation:
    """This is a class for make a simulation in order to evolve the state of the system.

    :param system: Object that contains index, position, type_, mu,
    anisotropy_constant, anisotopy_axis and field_axis (geometry). Also it contains a
    source, target, and jex (neighbors). Finally it contains units, damping,
    gyromagnetic, and deltat.
    :type system:
    :param temperature: The temperature of the sites in the system.
    :type temperature: float/list
    :param field: The field that acts under the sites in the system.
    :type field: float/list
    :param num_iterations: The number of iterations for evolve the system.
    :type num_iterations: int
    :param seed: The seed for the random state.
    :type seed: int
    :param initial_state: The initial state of the sites in te system.
    :type initial_state: list
    """

    def __init__(
        self,
        system,
        temperature: Bucket,
        field: Bucket,
        num_iterations=None,
        seed=None,
        initial_state=None,
    ):
        """
        The constructor for Simulation class.
        """
        self.system = system
        self.temperature = temperature
        self.field = field
        self.seed = seed

        if num_iterations:
            self.num_iterations = num_iterations
        else:
            self.num_iterations = 1000

        if seed:
            self.seed = seed
        else:
            self.seed = random.getrandbits(32)

        numpy.random.seed(self.seed)
        if initial_state:
            self.initial_state = initial_state
        else:
            self.initial_state = get_random_state(self.system.geometry.num_sites)

        self.initial_state = numpy.array(self.initial_state)

  
    def set_num_iterations(self, num_iterations):
        """It is a function to set the number of iterations.

        :param num_iterations: The number of iterations for evolve the system.
        :type num_iterations: int
        """
        self.num_iterations = num_iterations

    def set_initial_state(self, initial_state):
        """It is a function to set the initial state for each site.

        :param initial_state: The initial state of the sites in te system.
        :type initial_state: list
        """
        self.initial_state = numpy.array(initial_state)

    @property
    def information(self):
        """It is a function decorator, it creates an object with the complete
        information needed for the ``run`` function.


        """
        return {
            "num_sites": self.system.geometry.num_sites,
            "parameters": self.system.parameters,
            "temperature": self.temperature.values,
            "field": self.field.values,
            "seed": self.seed,
            "num_iterations": self.num_iterations,
            "positions": self.system.geometry.positions,
            "types": self.system.geometry.types,
            "initial_state": self.initial_state,
            "num_TH": len(self.temperature),
        }

    def run(self):
        """This function creates a generator. It calculates the evolve of the states
        through the implementation of the LLG equation. Also, it uses these states for
        calculate the exchange energy, anisotropy energy, magnetic energy, and hence,
        the total energy of the system."""


        spin_norms = self.system.geometry.spin_norms
        damping = self.system.damping
        deltat = self.system.deltat
        gyromagnetic = self.system.gyromagnetic
        kb = self.system.kb
        field_axes = self.system.geometry.field_axes
        exchanges = self.system.geometry.exchanges
        neighbors = self.system.geometry.neighbors
        anisotropy_constants = self.system.geometry.anisotropy_constants
        anisotropy_vectors = self.system.geometry.anisotropy_axes
        num_sites = self.system.geometry.num_sites
        state = self.initial_state

        for T, H in zip(self.temperature, self.field):
            temperatures = numpy.array([T] * num_sites)
            magnetic_fields = H * field_axes

            for _ in (range(self.num_iterations)):
                state = integrate(
                    state,
                    spin_norms,
                    temperatures,
                    damping,
                    deltat,
                    gyromagnetic,
                    kb,
                    magnetic_fields,
                    exchanges,
                    neighbors,
                    anisotropy_constants,
                    anisotropy_vectors,
                )

                exchange_energy_value = compute_exchange_energy(
                    state, exchanges, neighbors
                )
                anisotropy_energy_value = compute_anisotropy_energy(
                    state, anisotropy_constants, anisotropy_vectors
                )
                magnetic_energy_value = compute_magnetic_energy(
                    state, spin_norms, magnetic_fields
                )
                total_energy_value = (
                    exchange_energy_value
                    + anisotropy_energy_value
                    + magnetic_energy_value
                )

                yield (
                    state,
                    exchange_energy_value,
                    anisotropy_energy_value,
                    magnetic_energy_value,
                    total_energy_value,
                )