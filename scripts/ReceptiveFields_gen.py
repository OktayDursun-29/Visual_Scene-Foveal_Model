from genjax import ExactDensity
from genjax import Pytree

from ReceptiveFieldsJAX import rf_random, rf_logpdf


@Pytree.dataclass
class ReceptiveFields(ExactDensity):

    def sample(self, key, scene, rfs):
        return rf_random(
            key,
            scene,
            rfs
        )

    def logpdf(self, value, scene, rfs):
        return rf_logpdf(
            value,
            scene,
            rfs
        )