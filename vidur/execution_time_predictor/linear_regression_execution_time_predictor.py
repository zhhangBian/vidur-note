from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from vidur.config import (
    BaseReplicaSchedulerConfig,
    LinearRegressionExecutionTimePredictorConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)


# 使用线性回归的多项式方式进行时间预测
class LinearRegressionExecutionTimePredictor(SklearnExecutionTimePredictor):
    def __init__(
        self,
        predictor_config: LinearRegressionExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        # will trigger model training
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
        )

    def _get_grid_search_params(self):
        return {
            "polynomialfeatures__degree": self._config.polynomial_degree,
            "polynomialfeatures__include_bias": self._config.polynomial_include_bias,
            "polynomialfeatures__interaction_only": self._config.polynomial_interaction_only,
            "linearregression__fit_intercept": self._config.fit_intercept,
        }

    def _get_estimator(self):
        # 创建一个包含多步操作的管道
        # 1. 将输入特征转换为多项式特征
        # 2. 在多项式特征空间中找到最佳拟合线
        return make_pipeline(PolynomialFeatures(), LinearRegression())
