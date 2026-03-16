"""Fixed bridge record types for Python -> Mojo SPIRAL inference."""


@value
struct SpiralBridgeRecord:
    var descriptor_consistency: Float64
    var conformer_energy_variance: Float64
    var rmsd_cluster_density: Float64
    var torsion_entropy: Float64
    var shape_similarity_index: Float64
    var pharmacophore_deviation: Float64
    var minimization_success_rate: Float64
    var pipeline_instability_flags: Float64
    var protein_branch: Bool
    var missing_signal_count: Int


fn bridge_to_signal(record: SpiralBridgeRecord) -> SpiralSignalVector:
    return SpiralSignalVector(
        record.descriptor_consistency,
        record.conformer_energy_variance,
        record.rmsd_cluster_density,
        record.torsion_entropy,
        record.shape_similarity_index,
        record.pharmacophore_deviation,
        record.minimization_success_rate,
        record.pipeline_instability_flags,
    )


fn bridge_penalty(record: SpiralBridgeRecord) -> Float64:
    var penalty = 0.0
    if record.protein_branch:
        penalty += 0.05
    penalty += Float64(record.missing_signal_count) * 0.03
    return penalty
