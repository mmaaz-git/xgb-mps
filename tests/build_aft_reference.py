"""Extract upstream AFT expressions, substituting only math/distribution callbacks.

The generated test header replays GPU intermediates through unmodified source
arithmetic. It is not a copy of the Metal formula and not NVIDIA execution.
"""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def struct_source(source, name):
    start = source.index(f"struct {name} {{")
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    assert source[end] == ";"
    return source[start:end + 1]


def main():
    common = ROOT / "xgboost/src/common"
    probability = (common / "probability_distribution.h").read_text()
    survival = (common / "survival_util.h").read_text()
    parts = ["// Generated from upstream source by tests/build_aft_reference.py.",
             "// SHA256 probability/survival: " + hashlib.sha256(probability.encode()).hexdigest()
             + "/" + hashlib.sha256(survival.encode()).hexdigest(),
             "#pragma once", '#include "xgboost/base.h"',
             '#include "../xgboost/src/common/survival_util.h"', "namespace aft_replay {",
             "using std::sqrt; using std::log; using std::fmax; using std::isinf; using std::isnan;",
             "using xgboost::common::ProbabilityDistributionType;",
             "using xgboost::common::CensoringType; using xgboost::common::kPI;",
             "namespace aft = xgboost::common::aft;"]
    for name in ("NormalDistribution", "LogisticDistribution", "ExtremeDistribution"):
        body = struct_source(probability, name)
        body = body.replace("exp(", "Math::Exp(").replace("erf(", "Math::Erf(")
        parts.append("template <typename Math>\n" + body)
    body = struct_source(survival, "AFTLoss")
    for bound in ("lower", "upper"):
        term = f"log(y_{bound})"
        assert body.count(term) == 3
        body = body.replace(term, f"Distribution::Log(y_{bound})")
    for method in ("Grad", "Hess"):
        term = f"GetLimit{method}AtInfPred<Distribution>"
        assert body.count(term) == 1
        body = body.replace(term, f"GetLimit{method}AtInfPred<typename Distribution::Source>")
    parts.extend(["template <typename Distribution>\n" + body, "}  // namespace aft_replay", ""])
    target = ROOT / "results/cuda_aft_reference_generated.h"
    target.parent.mkdir(exist_ok=True)
    target.write_text("\n".join(parts))
    print(f"Generated {target.relative_to(ROOT)} from upstream probability/AFT expressions")


if __name__ == "__main__":
    main()
