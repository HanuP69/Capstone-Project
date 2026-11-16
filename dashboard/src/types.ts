export type Decision = 'PASS' | 'FLAG' | 'BLOCK';
export type DatasetMode = '[Synthetic Benchmark]';

export interface ClaimVerifyResponse {
  claim_id: string;
  claim_text: string;
  category: string;
  F_score: number;
  P_score: number;
  D_score: number;
  G_score: number;
  decision: Decision;
  predicted_label: string;
  fact_explanation: string;
  policy_explanation: string;
  retrieved_contexts: string[];
  regenerated_claim?: string | null;
  weighting_mode: string;
  retrieval_method: string;
  is_synthetic_benchmark: boolean;
}
export interface QueryResponse {
  query: string;
  draft_response: string;
  final_grounded_answer: string;
  overall_groundedness: number;
  total_claims: number;
  passed_claims: number;
  flagged_claims: number;
  blocked_claims: number;
  latency_seconds: number;
  claims: ClaimVerifyResponse[];
  dataset_mode: string;
}
export interface AuditRow {
  id: string; category: string; claim: string; label: string;
  F: number; P: number; D: number; G: number; decision: Decision;
  pred: string; fact_explanation: string; policy_explanation: string;
  regenerated_claim: string; dataset_mode: string;
}
export interface MetricRow {
  Condition: string; Dataset: string; Sample_Size: number; Accuracy: number;
  Precision: number; Recall: number; F1: number; 'Hallucination_Rate_%': number; Latency_s: number;
}
export interface MfaqConditionRow {
  Condition: string; Dataset: string; Sample_Size: number; Accuracy: number;
  Precision: number; Recall: number; F1_Score: number; 'Hallucination_Rate_%': number; Latency_s: number;
}
export interface RetrievalRow {
  'Recall@1': number; 'Recall@3': number; 'Recall@5': number; 'Recall@10': number;
  MRR: number; Method: string; Latency_ms: number;
}
export interface WeightRow {
  Weighting_Method: string; w1_Fact: number; w2_Policy: number; w3_Grounding: number;
  Derivation_Basis: string; NITJ_Accuracy: number; PublicUniv_Accuracy: number;
  MFAQ_Accuracy: number; Cross_Dataset_Mean_Accuracy: number;
}
export interface HumanEval {
  sample_size: number; dataset_mode: string; annotator_1_vs_annotator_2_kappa: number;
  kappa_target_met: boolean; inter_annotator_raw_agreement_pct: number;
  pipeline_to_human_consensus_agreement_pct: number;
}
export interface EvidenceDoc {
  id: string; title: string; category: string; is_synthetic: boolean; text: string;
}
export interface DataBundle {
  metrics: MetricRow[]; primaryAudit: AuditRow[]; secondaryAudit: AuditRow[];
  mfaqComparison: MfaqConditionRow[]; retrievalComparison: RetrievalRow[];
  weightsComparison: WeightRow[]; mfaqAudit: Record<string, unknown>[]; humanEval: HumanEval; documents: EvidenceDoc[];
}
