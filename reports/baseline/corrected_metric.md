# Wing 1.6: Corrected Baseline & Dangerous Error Definition

## Bug in Wing 1.5
The original `has_dangerous_error` metric incorrectly flagged any (pred, gt) pair with IoU<0.5. 
It should flag based on each prediction's BEST matching GT. This inflated error rates on 
high-mAP classes like `patches` (mAP50=0.916 but reported 93% error rate).

## Corrected Definition
A prediction is a HIGH-CONF FALSE POSITIVE only if its BEST IoU with any same-class GT is <0.5 AND score >0.5.
An image is DANGEROUS if it has >=1 high-conf FP OR >=1 FN.

## Implication
All routing analysis must use the corrected metric. Wing 1.5's negative result stands 
but must be re-verified with the corrected data.
