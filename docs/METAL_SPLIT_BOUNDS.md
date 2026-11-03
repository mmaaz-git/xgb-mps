# Conservative screening for scalar split evaluation

The optimization removes expensive software-binary64 gain calculations only
when an interval proves a candidate cannot win. Surviving candidates still use
`xgb_consider_split`, including the original float leaf weights, float squares,
binary64 intermediates, final float gain and CPU enumeration order for ties.

It applies to numerical features, including missing directions, with zero L1 penalty,
zero maximum leaf step, no monotonic constraint and infinite weight bounds.
Other cases retain the existing evaluator. Zero checks on the regularization
parameters inspect IEEE bits so a subnormal maximum step cannot be mistaken for
zero.

For a child with gradient `g`, Hessian `h` and L2 penalty `lambda`, write
`a = h + lambda` and `Q = g²/a`. For the exact optimum, the quadratic gain is
`Q`. If the computed float weight is `w = -(g/a)(1+δ)`, its unrounded quadratic
gain is `Q(1-δ²)`. The float square contributes an additional relative error
bounded by `(1+δ)² * ε`, followed by binary64 arithmetic and final float rounding.
For the normal, finite arithmetic range used here, those terms together are
less than `4ε`, where `ε = 2^-23`. The implementation uses a wider `32ε` margin.

The input gradient and Hessian are narrowed with the existing software conversion.
Their adjacent floats bracket the original binary64 values. Positive interval
arithmetic then brackets `Q`: square the gradient endpoints outward, add the
penalty to Hessian endpoints outward, and divide using opposite denominator
endpoints. The `32ε` margin covers the CPU weight/square calculation, and outward
float additions and subtraction include the parent gain and final rounding.

This path is restricted to rounded nonzero gradient magnitudes and positive
Hessians in `[1e-8, 1e8]`, and penalties in `[0, 1e8]`. These limits keep the
intermediate squares, denominator, quotient, weight and float weight square
normal and finite. Exact-zero gradients have zero child gain. Outside these
conditions the interval is unbounded, so the candidate receives full evaluation.
The outward-rounding helpers widen values near zero to normal endpoints to avoid
Metal's permitted flushing of subnormals.

These bounds rely on the correctly rounded basic operations with fast math
disabled specified in Apple's [Metal Shading Language Specification, sections
8.1, 8.2 and 8.4](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
One adjacent float covers either permitted rounding direction. Reassociation and
fast math remain disabled in production and in the tests.

The kernel takes the maximum lower bound among valid candidates and zero (the
original positive-gain acceptance threshold). It rejects a candidate only when
its upper bound is **strictly less** than that cutoff. A true winning candidate
or tied winner therefore remains eligible for the unchanged precise evaluator.
For features without missing rows, bounds occupy the unused suffix portion of
the scan buffer. Features with missing rows recompute upper bounds from their
forward/backward scans during screening. No category scratch allocation is assumed.

Validation includes one million independent CPU gain-containment checks, exact
CPU comparison of split choices/gains/child sums, near ties, equal gains,
restricted and unrestricted ranges, and full training comparisons. A paired
kernel benchmark compiles both enabled and disabled versions from the same
source and checks both against the CPU before recording timings.
