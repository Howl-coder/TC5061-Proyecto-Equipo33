import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE


def test_clustering_p3_es_reproducible_con_la_misma_semilla():
    rng = np.random.default_rng(3)
    x = np.vstack([rng.normal(loc=c, size=(30, 4)) for c in (-3, 0, 3)])

    def agrupar():
        # Misma configuración que el pipeline final de src.train_clustering
        pipeline = make_pipeline(
            StandardScaler(), KMeans(n_clusters=3, n_init=1, random_state=RANDOM_STATE)
        ).fit(x)
        return pipeline.predict(x), pipeline[-1].inertia_

    etiquetas_1, inercia_1 = agrupar()
    etiquetas_2, inercia_2 = agrupar()

    np.testing.assert_array_equal(etiquetas_1, etiquetas_2)
    assert inercia_1 == inercia_2
