from recbole.evaluator.base_metric import *
from recbole.evaluator.metrics import *
from recbole.evaluator.evaluator import *
import recbole.evaluator.register as register
from recbole.evaluator.collector import *
import recbole_debias.evaluator.metrics


def update_metrics():
    metric_module_name = metrics.__name__
    smaller_metrics, metric_information, metric_types, metrics_dict = register.cluster_info(metric_module_name)

    register.smaller_metrics += smaller_metrics
    register.metric_information.update(metric_information)
    register.metric_types.update(metric_types)
    register.metrics_dict.update(metrics_dict)
