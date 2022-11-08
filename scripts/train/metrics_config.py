from torchmetrics import Accuracy, JaccardIndex, Precision, Specificity

from apriorics.metrics import (
    DetectionSegmentationMetrics,
    DiceScore,
    Recall,
    SegmentationAUC,
)

METRICS = {
    "all": [
        JaccardIndex(2),
        DiceScore(),
        Accuracy(),
        Precision(),
        Recall(),
        Specificity(),
        SegmentationAUC(),
    ],
    "PHH3": [DetectionSegmentationMetrics(flood_fill=True)],
}
