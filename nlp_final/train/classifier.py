from pathlib import Path
from typing import Any, List

from numpy import ndarray

# from
# def trainMultinomialNaiveBayes(
#     x: ndarray, y: ndarray, outputPath: Path
# ) -> None:
#     parameters: dict[str, List[Any]] = {
#         "multinomialnb__alpha": [1.0, 0.5, 1.5],
#         "multinomialnb__force_alpha": [True],
#     }

#     pipeline: Pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
#     gscv: GridSearchCV = GridSearchCV(
#         estimator=pipeline, param_grid=parameters, n_jobs=1
#     )
#     gscv.fit(X=x, y=y)
#     model: Pipeline = gscv.best_estimator_
#     dump(value=model, filename=Path(outputPath, "multinomialNB.joblib"))
