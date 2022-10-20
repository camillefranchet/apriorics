from torchmetrics import Accuracy, JaccardIndex, Precision, Recall, Specificity

from apriorics.metrics import DetectionSegmentationMetrics, DiceScore, SegmentationAUC

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
    "PHH3": [DetectionSegmentationMetrics()],
}
