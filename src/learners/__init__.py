from .lica_learner import LICALearner
from .tc_learner import TCLearner
from .attn_learner import ATTNLearner
from .tclica_learner import TCLICALearner

REGISTRY = {}

REGISTRY["lica_learner"] = LICALearner
REGISTRY["tc_learner"] = TCLearner
REGISTRY["attn_learner"] = ATTNLearner
REGISTRY["tclica_learner"] = TCLICALearner
