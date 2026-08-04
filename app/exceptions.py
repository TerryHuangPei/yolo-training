class PipelineError(Exception):
    """Base exception for predictable pipeline failures."""


class DatasetExtractionError(PipelineError):
    pass


class DatasetFormatError(PipelineError):
    pass


class DatasetValidationError(PipelineError):
    pass


class TrainingError(PipelineError):
    pass


class EvaluationError(PipelineError):
    pass


class InferenceError(PipelineError):
    pass


class ArtifactPromotionError(PipelineError):
    pass


class JobStateError(PipelineError):
    pass
