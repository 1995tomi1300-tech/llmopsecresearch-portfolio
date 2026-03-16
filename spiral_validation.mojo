"""SPIRAL validation engine in Mojo.

This module implements a deterministic trust and anomaly layer for scientific
pipelines. It does not repair chemistry and does not simulate physics. It
consumes normalized upstream signals and emits explainable audit decisions.
"""

from std.collections import List


# ===-----------------------------------------------------------------------===#
# Constants
# ===-----------------------------------------------------------------------===#


comptime ANOM_RMSD_TORSION = 1
comptime ANOM_MIN_PHARMA = 2
comptime ANOM_DESCRIPTOR_STRUCTURE = 4
comptime ANOM_ENERGY_CLUSTER = 8
comptime ANOM_EXECUTION_INSTABILITY = 16

comptime REGIME_STABLE = 0
comptime REGIME_QUESTIONABLE = 1
comptime REGIME_DISCARD = 2

comptime ACTION_ACCEPT = 0
comptime ACTION_MANUAL_REVIEW = 1
comptime ACTION_RERUN_PIPELINE = 2
comptime ACTION_RERUN_MINIMIZATION = 3
comptime ACTION_RERUN_CONFORMERS = 4
comptime ACTION_RERUN_DESCRIPTORS = 5
comptime ACTION_DISCARD = 6


# ===-----------------------------------------------------------------------===#
# Utility functions
# ===-----------------------------------------------------------------------===#


def clamp01(x: Float64) -> Float64:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def inverse_scaled(x: Float64, scale: Float64) -> Float64:
    var safe = x
    if safe < 0.0:
        safe = 0.0
    return 1.0 / (1.0 + safe / scale)


def bit_count(mask: Int) -> Int:
    var value = mask
    var count = 0
    while value > 0:
        count += value & 1
        value = value >> 1
    return count


def has_flag(mask: Int, flag: Int) -> Bool:
    return (mask & flag) != 0


def weighted_mean(
    lhs_value: Float64,
    lhs_weight: Float64,
    rhs_value: Float64,
    rhs_weight: Float64,
) -> Float64:
    var total = lhs_weight + rhs_weight
    if total <= 0.0:
        return 0.0
    return (lhs_value * lhs_weight + rhs_value * rhs_weight) / total


def structural_stability(
    conformer_good: Float64,
    rmsd_good: Float64,
    torsion_good: Float64,
    shape_good: Float64,
    pharmacophore_good: Float64,
) -> Float64:
    return (
        conformer_good
        + rmsd_good
        + torsion_good
        + shape_good
        + pharmacophore_good
    ) / 5.0


# ===-----------------------------------------------------------------------===#
# Signal model
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct SpiralSignalVector(ImplicitlyCopyable):
    var descriptor_consistency: Float64
    var conformer_energy_variance: Float64
    var rmsd_cluster_density: Float64
    var torsion_entropy: Float64
    var shape_similarity_index: Float64
    var pharmacophore_deviation: Float64
    var minimization_success_rate: Float64
    var pipeline_instability_flags: Float64


@fieldwise_init
struct SpiralDecision(ImplicitlyCopyable):
    var trust_score: Float64
    var false_positive_risk: Float64
    var anomaly_mask: Int
    var anomaly_count: Int
    var regime_code: Int
    var rerun_code: Int
    var cohort_alignment: Float64
    var memory_size: Int


@fieldwise_init
struct SpiralWeights(ImplicitlyCopyable):
    var descriptor: Float64
    var conformer: Float64
    var rmsd: Float64
    var torsion: Float64
    var shape: Float64
    var pharmacophore: Float64
    var minimization: Float64
    var instability: Float64


@fieldwise_init
struct SpiralCohortSignal(ImplicitlyCopyable):
    var logical_node_count: Float64
    var descriptor_consistency: Float64
    var conformer_energy_variance: Float64
    var rmsd_cluster_density: Float64
    var torsion_entropy: Float64
    var shape_similarity_index: Float64
    var pharmacophore_deviation: Float64
    var minimization_success_rate: Float64
    var pipeline_instability_flags: Float64


@fieldwise_init
struct SpiralBatchSummary(ImplicitlyCopyable):
    var cohort_count: Int
    var stable_count: Int
    var questionable_count: Int
    var discard_count: Int
    var anomaly_count: Int
    var logical_node_total: Float64
    var mean_trust_score: Float64
    var mean_false_positive_risk: Float64
    var mean_cohort_alignment: Float64


def default_weights() -> SpiralWeights:
    return SpiralWeights(0.18, 0.14, 0.14, 0.12, 0.12, 0.12, 0.10, 0.08)


def cohort_to_signal(cohort: SpiralCohortSignal) -> SpiralSignalVector:
    return SpiralSignalVector(
        cohort.descriptor_consistency,
        cohort.conformer_energy_variance,
        cohort.rmsd_cluster_density,
        cohort.torsion_entropy,
        cohort.shape_similarity_index,
        cohort.pharmacophore_deviation,
        cohort.minimization_success_rate,
        cohort.pipeline_instability_flags,
    )


# ===-----------------------------------------------------------------------===#
# Core evaluation
# ===-----------------------------------------------------------------------===#


def detect_anomalies(signal: SpiralSignalVector) -> Int:
    var mask = 0

    if signal.rmsd_cluster_density >= 0.65 and signal.torsion_entropy >= 0.55:
        mask = mask | ANOM_RMSD_TORSION

    if (
        signal.minimization_success_rate >= 0.80
        and signal.pharmacophore_deviation >= 0.55
    ):
        mask = mask | ANOM_MIN_PHARMA

    var conformer_good = clamp01(
        inverse_scaled(signal.conformer_energy_variance, 0.20)
    )
    var torsion_good = clamp01(1.0 - signal.torsion_entropy)
    var shape_good = clamp01(signal.shape_similarity_index)
    var pharmacophore_good = clamp01(
        inverse_scaled(signal.pharmacophore_deviation, 0.60)
    )
    var structure_good = structural_stability(
        conformer_good,
        clamp01(signal.rmsd_cluster_density),
        torsion_good,
        shape_good,
        pharmacophore_good,
    )

    if signal.descriptor_consistency >= 0.75 and structure_good <= 0.45:
        mask = mask | ANOM_DESCRIPTOR_STRUCTURE

    if (
        signal.conformer_energy_variance >= 0.55
        and signal.rmsd_cluster_density >= 0.65
    ):
        mask = mask | ANOM_ENERGY_CLUSTER

    if (
        signal.minimization_success_rate >= 0.70
        and signal.pipeline_instability_flags >= 0.50
    ):
        mask = mask | ANOM_EXECUTION_INSTABILITY

    return mask


def classify_regime(
    trust_score: Float64, anomaly_mask: Int, signal: SpiralSignalVector
) -> Int:
    if (
        has_flag(anomaly_mask, ANOM_DESCRIPTOR_STRUCTURE)
        or signal.pipeline_instability_flags >= 0.75
    ):
        return REGIME_DISCARD
    if trust_score >= 0.75:
        return REGIME_STABLE
    if trust_score >= 0.45:
        return REGIME_QUESTIONABLE
    return REGIME_DISCARD


def recommend_action(
    regime_code: Int, signal: SpiralSignalVector, anomaly_mask: Int
) -> Int:
    if regime_code == REGIME_STABLE and anomaly_mask == 0:
        return ACTION_ACCEPT
    if signal.pipeline_instability_flags >= 0.60:
        return ACTION_RERUN_PIPELINE
    if signal.minimization_success_rate < 0.45:
        return ACTION_RERUN_MINIMIZATION
    if (
        signal.torsion_entropy >= 0.60
        or signal.conformer_energy_variance >= 0.60
    ):
        return ACTION_RERUN_CONFORMERS
    if signal.descriptor_consistency < 0.45:
        return ACTION_RERUN_DESCRIPTORS
    if regime_code == REGIME_DISCARD:
        return ACTION_DISCARD
    return ACTION_MANUAL_REVIEW


def evaluate(signal: SpiralSignalVector) -> SpiralDecision:
    var weights = default_weights()

    var descriptor_good = clamp01(signal.descriptor_consistency)
    var conformer_good = clamp01(
        inverse_scaled(signal.conformer_energy_variance, 0.20)
    )
    var rmsd_good = clamp01(signal.rmsd_cluster_density)
    var torsion_good = clamp01(1.0 - signal.torsion_entropy)
    var shape_good = clamp01(signal.shape_similarity_index)
    var pharmacophore_good = clamp01(
        inverse_scaled(signal.pharmacophore_deviation, 0.60)
    )
    var minimization_good = clamp01(signal.minimization_success_rate)
    var instability_good = clamp01(1.0 - signal.pipeline_instability_flags)

    var anomaly_mask = detect_anomalies(signal)
    var anomaly_count = bit_count(anomaly_mask)

    var trust_score = (
        weights.descriptor * descriptor_good
        + weights.conformer * conformer_good
        + weights.rmsd * rmsd_good
        + weights.torsion * torsion_good
        + weights.shape * shape_good
        + weights.pharmacophore * pharmacophore_good
        + weights.minimization * minimization_good
        + weights.instability * instability_good
    )

    if has_flag(anomaly_mask, ANOM_DESCRIPTOR_STRUCTURE):
        trust_score -= 0.12

    var extra_anomalies = anomaly_count
    if has_flag(anomaly_mask, ANOM_DESCRIPTOR_STRUCTURE):
        extra_anomalies -= 1
    trust_score -= Float64(extra_anomalies) * 0.06
    trust_score = clamp01(trust_score)

    var false_positive_risk = (
        0.60 * (1.0 - trust_score)
        + 0.25
            * clamp01(
                0.20 * Float64(anomaly_count)
                + 0.40 * signal.pipeline_instability_flags
            )
        + 0.15 * signal.pipeline_instability_flags
    )
    false_positive_risk = clamp01(false_positive_risk)

    var regime_code = classify_regime(trust_score, anomaly_mask, signal)
    var rerun_code = recommend_action(regime_code, signal, anomaly_mask)

    return SpiralDecision(
        trust_score,
        false_positive_risk,
        anomaly_mask,
        anomaly_count,
        regime_code,
        rerun_code,
        0.50,
        0,
    )


# ===-----------------------------------------------------------------------===#
# SpiralAIEngine
# ===-----------------------------------------------------------------------===#


struct SpiralAIEngine:
    var memory_count: Int
    var stable_memory_count: Int
    var mean_descriptor_consistency: Float64
    var mean_conformer_energy_variance: Float64
    var mean_rmsd_cluster_density: Float64
    var mean_torsion_entropy: Float64
    var mean_shape_similarity_index: Float64
    var mean_pharmacophore_deviation: Float64
    var mean_minimization_success_rate: Float64
    var mean_pipeline_instability_flags: Float64

    def __init__(out self):
        self.memory_count = 0
        self.stable_memory_count = 0
        self.mean_descriptor_consistency = 0.0
        self.mean_conformer_energy_variance = 0.0
        self.mean_rmsd_cluster_density = 0.0
        self.mean_torsion_entropy = 0.0
        self.mean_shape_similarity_index = 0.0
        self.mean_pharmacophore_deviation = 0.0
        self.mean_minimization_success_rate = 0.0
        self.mean_pipeline_instability_flags = 0.0

    def cohort_alignment(self, signal: SpiralSignalVector) -> Float64:
        if self.stable_memory_count == 0:
            return 0.50

        var distance = (
            abs(signal.descriptor_consistency - self.mean_descriptor_consistency)
            + abs(
                signal.conformer_energy_variance
                - self.mean_conformer_energy_variance
            )
            + abs(signal.rmsd_cluster_density - self.mean_rmsd_cluster_density)
            + abs(signal.torsion_entropy - self.mean_torsion_entropy)
            + abs(
                signal.shape_similarity_index
                - self.mean_shape_similarity_index
            )
            + abs(
                signal.pharmacophore_deviation
                - self.mean_pharmacophore_deviation
            )
            + abs(
                signal.minimization_success_rate
                - self.mean_minimization_success_rate
            )
            + abs(
                signal.pipeline_instability_flags
                - self.mean_pipeline_instability_flags
            )
        ) / 8.0

        return clamp01(1.0 - distance)

    def update_memory(mut self, signal: SpiralSignalVector, decision: SpiralDecision):
        self.memory_count += 1
        if decision.regime_code != REGIME_STABLE:
            return

        var count = Float64(self.stable_memory_count)
        var denom = count + 1.0

        self.mean_descriptor_consistency = (
            self.mean_descriptor_consistency * count
            + signal.descriptor_consistency
        ) / denom
        self.mean_conformer_energy_variance = (
            self.mean_conformer_energy_variance * count
            + signal.conformer_energy_variance
        ) / denom
        self.mean_rmsd_cluster_density = (
            self.mean_rmsd_cluster_density * count
            + signal.rmsd_cluster_density
        ) / denom
        self.mean_torsion_entropy = (
            self.mean_torsion_entropy * count + signal.torsion_entropy
        ) / denom
        self.mean_shape_similarity_index = (
            self.mean_shape_similarity_index * count
            + signal.shape_similarity_index
        ) / denom
        self.mean_pharmacophore_deviation = (
            self.mean_pharmacophore_deviation * count
            + signal.pharmacophore_deviation
        ) / denom
        self.mean_minimization_success_rate = (
            self.mean_minimization_success_rate * count
            + signal.minimization_success_rate
        ) / denom
        self.mean_pipeline_instability_flags = (
            self.mean_pipeline_instability_flags * count
            + signal.pipeline_instability_flags
        ) / denom

        self.stable_memory_count += 1

    def infer(mut self, signal: SpiralSignalVector) -> SpiralDecision:
        var base = evaluate(signal)
        var alignment = self.cohort_alignment(signal)

        var trust_score = base.trust_score
        var false_positive_risk = base.false_positive_risk
        var regime_code = base.regime_code

        if self.stable_memory_count > 0:
            trust_score = clamp01(0.85 * base.trust_score + 0.15 * alignment)
            false_positive_risk = clamp01(
                0.90 * base.false_positive_risk + 0.10 * (1.0 - alignment)
            )

            if (
                regime_code == REGIME_QUESTIONABLE
                and trust_score >= 0.78
                and base.anomaly_count == 0
            ):
                regime_code = REGIME_STABLE
            elif regime_code == REGIME_STABLE and alignment < 0.35:
                regime_code = REGIME_QUESTIONABLE

        var rerun_code = recommend_action(regime_code, signal, base.anomaly_mask)
        var result = SpiralDecision(
            trust_score,
            false_positive_risk,
            base.anomaly_mask,
            base.anomaly_count,
            regime_code,
            rerun_code,
            alignment,
            self.stable_memory_count,
        )

        self.update_memory(signal, result)
        return result


# ===-----------------------------------------------------------------------===#
# SpiralPopulationEngine
# ===-----------------------------------------------------------------------===#


struct SpiralPopulationEngine:
    var cohort_count: Int
    var logical_node_total: Float64
    var stable_logical_node_total: Float64
    var mean_descriptor_consistency: Float64
    var mean_conformer_energy_variance: Float64
    var mean_rmsd_cluster_density: Float64
    var mean_torsion_entropy: Float64
    var mean_shape_similarity_index: Float64
    var mean_pharmacophore_deviation: Float64
    var mean_minimization_success_rate: Float64
    var mean_pipeline_instability_flags: Float64

    def __init__(out self):
        self.cohort_count = 0
        self.logical_node_total = 0.0
        self.stable_logical_node_total = 0.0
        self.mean_descriptor_consistency = 0.0
        self.mean_conformer_energy_variance = 0.0
        self.mean_rmsd_cluster_density = 0.0
        self.mean_torsion_entropy = 0.0
        self.mean_shape_similarity_index = 0.0
        self.mean_pharmacophore_deviation = 0.0
        self.mean_minimization_success_rate = 0.0
        self.mean_pipeline_instability_flags = 0.0

    def cohort_alignment(self, cohort: SpiralCohortSignal) -> Float64:
        if self.stable_logical_node_total <= 0.0:
            return 0.50

        var distance = (
            abs(cohort.descriptor_consistency - self.mean_descriptor_consistency)
            + abs(
                cohort.conformer_energy_variance
                - self.mean_conformer_energy_variance
            )
            + abs(cohort.rmsd_cluster_density - self.mean_rmsd_cluster_density)
            + abs(cohort.torsion_entropy - self.mean_torsion_entropy)
            + abs(
                cohort.shape_similarity_index
                - self.mean_shape_similarity_index
            )
            + abs(
                cohort.pharmacophore_deviation
                - self.mean_pharmacophore_deviation
            )
            + abs(
                cohort.minimization_success_rate
                - self.mean_minimization_success_rate
            )
            + abs(
                cohort.pipeline_instability_flags
                - self.mean_pipeline_instability_flags
            )
        ) / 8.0

        return clamp01(1.0 - distance)

    def update_memory(mut self, cohort: SpiralCohortSignal, decision: SpiralDecision):
        self.cohort_count += 1
        self.logical_node_total += cohort.logical_node_count

        if decision.regime_code != REGIME_STABLE:
            return

        var stable_weight = self.stable_logical_node_total
        var cohort_weight = cohort.logical_node_count
        if cohort_weight < 0.0:
            cohort_weight = 0.0

        self.mean_descriptor_consistency = weighted_mean(
            self.mean_descriptor_consistency,
            stable_weight,
            cohort.descriptor_consistency,
            cohort_weight,
        )
        self.mean_conformer_energy_variance = weighted_mean(
            self.mean_conformer_energy_variance,
            stable_weight,
            cohort.conformer_energy_variance,
            cohort_weight,
        )
        self.mean_rmsd_cluster_density = weighted_mean(
            self.mean_rmsd_cluster_density,
            stable_weight,
            cohort.rmsd_cluster_density,
            cohort_weight,
        )
        self.mean_torsion_entropy = weighted_mean(
            self.mean_torsion_entropy,
            stable_weight,
            cohort.torsion_entropy,
            cohort_weight,
        )
        self.mean_shape_similarity_index = weighted_mean(
            self.mean_shape_similarity_index,
            stable_weight,
            cohort.shape_similarity_index,
            cohort_weight,
        )
        self.mean_pharmacophore_deviation = weighted_mean(
            self.mean_pharmacophore_deviation,
            stable_weight,
            cohort.pharmacophore_deviation,
            cohort_weight,
        )
        self.mean_minimization_success_rate = weighted_mean(
            self.mean_minimization_success_rate,
            stable_weight,
            cohort.minimization_success_rate,
            cohort_weight,
        )
        self.mean_pipeline_instability_flags = weighted_mean(
            self.mean_pipeline_instability_flags,
            stable_weight,
            cohort.pipeline_instability_flags,
            cohort_weight,
        )

        self.stable_logical_node_total += cohort_weight

    def infer(mut self, cohort: SpiralCohortSignal) -> SpiralDecision:
        var signal = cohort_to_signal(cohort)
        var base = evaluate(signal)
        var alignment = self.cohort_alignment(cohort)

        var trust_score = base.trust_score
        var false_positive_risk = base.false_positive_risk
        var regime_code = base.regime_code

        if self.stable_logical_node_total > 0.0:
            trust_score = clamp01(0.80 * base.trust_score + 0.20 * alignment)
            false_positive_risk = clamp01(
                0.88 * base.false_positive_risk + 0.12 * (1.0 - alignment)
            )

            if (
                regime_code == REGIME_QUESTIONABLE
                and trust_score >= 0.80
                and base.anomaly_count == 0
            ):
                regime_code = REGIME_STABLE
            elif regime_code == REGIME_STABLE and alignment < 0.30:
                regime_code = REGIME_QUESTIONABLE

        var rerun_code = recommend_action(regime_code, signal, base.anomaly_mask)
        var result = SpiralDecision(
            trust_score,
            false_positive_risk,
            base.anomaly_mask,
            base.anomaly_count,
            regime_code,
            rerun_code,
            alignment,
            self.cohort_count,
        )

        self.update_memory(cohort, result)
        return result

    def audit_batch(mut self, cohorts: List[SpiralCohortSignal]) -> SpiralBatchSummary:
        var stable_count = 0
        var questionable_count = 0
        var discard_count = 0
        var anomaly_count = 0
        var logical_node_total = 0.0
        var trust_total = 0.0
        var false_positive_total = 0.0
        var alignment_total = 0.0

        for i in range(len(cohorts)):
            var cohort = cohorts[i]
            var decision = self.infer(cohort)

            logical_node_total += cohort.logical_node_count
            trust_total += decision.trust_score
            false_positive_total += decision.false_positive_risk
            alignment_total += decision.cohort_alignment

            if decision.regime_code == REGIME_STABLE:
                stable_count += 1
            elif decision.regime_code == REGIME_QUESTIONABLE:
                questionable_count += 1
            else:
                discard_count += 1

            if decision.anomaly_count > 0:
                anomaly_count += 1

        var batch_size = len(cohorts)
        if batch_size == 0:
            return SpiralBatchSummary(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

        var denom = Float64(batch_size)
        return SpiralBatchSummary(
            batch_size,
            stable_count,
            questionable_count,
            discard_count,
            anomaly_count,
            logical_node_total,
            trust_total / denom,
            false_positive_total / denom,
            alignment_total / denom,
        )


# ===-----------------------------------------------------------------------===#
# Display helpers
# ===-----------------------------------------------------------------------===#


def regime_label(code: Int) -> String:
    if code == REGIME_STABLE:
        return String("stable")
    if code == REGIME_QUESTIONABLE:
        return String("questionable")
    return String("discard")


def rerun_label(code: Int) -> String:
    if code == ACTION_ACCEPT:
        return String("accept")
    if code == ACTION_MANUAL_REVIEW:
        return String("manual_review")
    if code == ACTION_RERUN_PIPELINE:
        return String("rerun_pipeline_stage")
    if code == ACTION_RERUN_MINIMIZATION:
        return String("rerun_minimization")
    if code == ACTION_RERUN_CONFORMERS:
        return String("rerun_conformer_generation")
    if code == ACTION_RERUN_DESCRIPTORS:
        return String("rerun_descriptor_validation")
    return String("discard_candidate")


def print_result(result: SpiralDecision):
    print("trust_score=", result.trust_score)
    print("false_positive_risk=", result.false_positive_risk)
    print("regime_label=", regime_label(result.regime_code))
    print("rerun_recommendation=", rerun_label(result.rerun_code))
    print("anomaly_mask=", result.anomaly_mask)
    print("anomaly_count=", result.anomaly_count)
    print("cohort_alignment=", result.cohort_alignment)
    print("memory_size=", result.memory_size)


def print_batch_summary(summary: SpiralBatchSummary):
    print("batch_cohort_count=", summary.cohort_count)
    print("batch_logical_node_total=", summary.logical_node_total)
    print("batch_stable_count=", summary.stable_count)
    print("batch_questionable_count=", summary.questionable_count)
    print("batch_discard_count=", summary.discard_count)
    print("batch_anomaly_count=", summary.anomaly_count)
    print("batch_mean_trust_score=", summary.mean_trust_score)
    print("batch_mean_false_positive_risk=", summary.mean_false_positive_risk)
    print("batch_mean_cohort_alignment=", summary.mean_cohort_alignment)


# ===-----------------------------------------------------------------------===#
# Demo
# ===-----------------------------------------------------------------------===#


def main():
    var engine = SpiralAIEngine()

    var signal_a = SpiralSignalVector(0.87, 0.35, 0.54, 0.04, 0.92, 0.20, 0.50, 0.00)
    print_result(engine.infer(signal_a))

    var signal_b = SpiralSignalVector(0.82, 0.28, 0.61, 0.06, 0.89, 0.24, 0.74, 0.02)
    print_result(engine.infer(signal_b))

    var population_engine = SpiralPopulationEngine()

    var macro_cohort = SpiralCohortSignal(
        1e25, 0.84, 0.31, 0.58, 0.07, 0.88, 0.22, 0.71, 0.03
    )
    print_result(population_engine.infer(macro_cohort))

    var cohorts = List[SpiralCohortSignal]()
    cohorts.append(
        SpiralCohortSignal(
            3.2e24, 0.86, 0.27, 0.63, 0.05, 0.90, 0.19, 0.79, 0.02
        )
    )
    cohorts.append(
        SpiralCohortSignal(
            4.1e24, 0.72, 0.48, 0.57, 0.16, 0.78, 0.33, 0.68, 0.09
        )
    )
    cohorts.append(
        SpiralCohortSignal(
            2.7e24, 0.44, 0.74, 0.32, 0.61, 0.55, 0.69, 0.38, 0.34
        )
    )

    print_batch_summary(population_engine.audit_batch(cohorts))
