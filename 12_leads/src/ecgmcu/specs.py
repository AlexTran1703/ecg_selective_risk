"""Human-readable description of each encoder, for the specifications table.

Kept beside the model definitions so a change to an architecture and a
change to the row describing it stay in the same place. Block counts are
counted from the built module in s7_manuscript.py rather than trusted.
"""

SPECS = {
    "tiny": {
        "name": "TinyCNN", "blocks": "4 conv",
        "operator": "plain Conv1D, stride 2",
        "note": "baseline: no residual, no separable convolution"},
    "resnet": {
        "name": "ResNet1D-Lite", "blocks": "4 residual",
        "operator": "residual Conv1D",
        "note": "widest channels, highest capacity"},
    "dscnn": {
        "name": "DS-CNN", "blocks": "stem + 3 separable",
        "operator": "depthwise + pointwise",
        "note": "the common microcontroller workhorse"},
    "mobilenet": {
        "name": "MobileNet1D", "blocks": "stem + 5 separable",
        "operator": "depthwise + pointwise",
        "note": "deeper separable stack, alternating stride"},
    "mbconv": {
        "name": "MBConv1D", "blocks": "stem + 6 inverted bottleneck",
        "operator": "inverted bottleneck, expand 4",
        "note": "expansion inflates activations in float32"},
    "tcn": {
        "name": "TCN-Lite", "blocks": "4 dilated",
        "operator": "dilated Conv1D, d = 1/2/4/8",
        "note": "wide receptive field without recurrence"},
}
