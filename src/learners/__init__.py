from .lica_learner import LICALearner
from .tc_learner import TCLearner

REGISTRY = {}

REGISTRY["lica_learner"] = LICALearner
REGISTRY["tc_learner"] = TCLearner

