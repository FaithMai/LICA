from .lica_learner import LICALearner
from .tc_learner import TCLearner
from .attn_learner import ATTNLearner

REGISTRY = {}

REGISTRY["lica_learner"] = LICALearner
REGISTRY["tc_learner"] = TCLearner
REGISTRY["attn_learner"] = ATTNLearner
