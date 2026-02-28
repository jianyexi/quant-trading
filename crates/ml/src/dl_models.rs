//! Deep Learning Multi-Factor Model Research Knowledge Base
//!
//! Curated collection of state-of-the-art DL models for quantitative factor
//! extraction and trading signal generation. Includes auto-collection capability
//! via LLM summarization.

use serde::{Deserialize, Serialize};

/// A deep learning factor model entry in the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlModelEntry {
    pub id: String,
    pub name: String,
    pub category: String,
    pub year: u16,
    pub architecture: String,
    pub description: String,
    pub key_innovation: String,
    pub input_data: Vec<String>,
    pub output: String,
    pub strengths: Vec<String>,
    pub limitations: Vec<String>,
    pub reference: String,
    pub reference_url: String,
}

/// Category for grouping DL models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCategory {
    pub name: String,
    pub description: String,
    pub models: Vec<DlModelEntry>,
}

/// Collected research item from LLM-assisted web collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectedResearch {
    pub title: String,
    pub summary: String,
    pub source: String,
    pub relevance: String,
    pub collected_at: String,
}

/// Research knowledge base with curated + collected models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchKnowledgeBase {
    pub categories: Vec<ModelCategory>,
    pub collected: Vec<CollectedResearch>,
    pub last_updated: String,
}

/// Build the curated knowledge base of DL factor models.
pub fn build_knowledge_base() -> ResearchKnowledgeBase {
    let categories = vec![
        ModelCategory {
            name: "Transformer系列".into(),
            description: "基于自注意力机制的序列建模，擅长捕捉长距离时序依赖和多因子交互".into(),
            models: vec![
                DlModelEntry {
                    id: "quantformer".into(),
                    name: "Quantformer".into(),
                    category: "Transformer".into(),
                    year: 2024,
                    architecture: "Enhanced Transformer + Sentiment Embedding".into(),
                    description: "专为量化交易设计的Transformer架构，融合情绪分析进行因子提取。在中国A股超过100个传统因子策略的对比中表现优异。".into(),
                    key_innovation: "将市场情绪编码为因子向量，与价量因子在注意力层中深度融合".into(),
                    input_data: vec!["多因子股票数据".into(), "情绪文本特征".into()],
                    output: "收益率预测 + 因子权重".into(),
                    strengths: vec![
                        "超越100+传统因子策略".into(),
                        "情绪因子与价量因子深度融合".into(),
                        "长序列依赖建模能力强".into(),
                    ],
                    limitations: vec![
                        "训练数据量要求大".into(),
                        "计算资源需求高".into(),
                        "黑箱特性影响可解释性".into(),
                    ],
                    reference: "Quantformer: from attention to profit (AAAI 2024)".into(),
                    reference_url: "https://huggingface.co/papers/2404.00424".into(),
                },
                DlModelEntry {
                    id: "tft_asro".into(),
                    name: "TFT-ASRO".into(),
                    category: "Transformer".into(),
                    year: 2025,
                    architecture: "Temporal Fusion Transformer + Adaptive Sharpe Ratio Optimization".into(),
                    description: "多传感器时序融合Transformer，直接优化风险调整收益(Sharpe比率)，支持多源数据输入。".into(),
                    key_innovation: "自适应Sharpe比率作为损失函数，而非传统MSE，使模型直接优化风险收益比".into(),
                    input_data: vec!["价量数据".into(), "基本面指标".into(), "情绪传感器".into()],
                    output: "风险调整后收益预测".into(),
                    strengths: vec![
                        "直接优化投资目标(Sharpe)".into(),
                        "多源数据融合能力".into(),
                        "波动市场表现优异".into(),
                    ],
                    limitations: vec![
                        "Sharpe优化可能过拟合特定市场状态".into(),
                        "多传感器标定需要领域知识".into(),
                    ],
                    reference: "Multi-Sensor TFT for Stock Performance Prediction (Sensors, 2025)".into(),
                    reference_url: "https://www.mdpi.com/1424-8220/25/3/976".into(),
                },
                DlModelEntry {
                    id: "informer".into(),
                    name: "Informer / Autoformer".into(),
                    category: "Transformer".into(),
                    year: 2024,
                    architecture: "ProbSparse Attention + Auto-Correlation".into(),
                    description: "高效长序列Transformer变体，通过稀疏注意力降低O(n²)复杂度，适合金融高频/长序列建模。".into(),
                    key_innovation: "ProbSparse自注意力机制，自动发现序列中的周期性模式".into(),
                    input_data: vec!["长时间序列价量数据".into()],
                    output: "多步预测序列".into(),
                    strengths: vec![
                        "O(n·log n)复杂度，支持超长序列".into(),
                        "自动捕捉周期性模式".into(),
                        "推理速度快".into(),
                    ],
                    limitations: vec![
                        "对非平稳金融数据可能欠拟合".into(),
                        "需要大量历史数据".into(),
                    ],
                    reference: "Informer (AAAI 2021) / Autoformer (NeurIPS 2021)".into(),
                    reference_url: "https://arxiv.org/abs/2012.07436".into(),
                },
            ],
        },
        ModelCategory {
            name: "VAE/生成式模型".into(),
            description: "基于变分自编码器的概率因子模型，擅长提取隐含风险因子和市场状态".into(),
            models: vec![
                DlModelEntry {
                    id: "factorvae".into(),
                    name: "FactorVAE".into(),
                    category: "VAE".into(),
                    year: 2024,
                    architecture: "Variational Autoencoder + Dynamic Factor Loading".into(),
                    description: "概率动态因子模型，基于VAE提取统计独立的隐含风险因子。相比OLS降低44%重构误差，能有效检测结构性市场变化。".into(),
                    key_innovation: "将传统因子模型概率化，因子载荷随时间动态变化，自动发现隐含风险结构".into(),
                    input_data: vec!["截面收益率数据".into(), "基本面因子".into()],
                    output: "隐含因子 + 动态因子载荷 + 风险度量".into(),
                    strengths: vec![
                        "比OLS降低44%重构误差".into(),
                        "COVID/政策变化等结构突变检测".into(),
                        "因子统计独立性保证".into(),
                        "提供不确定性量化".into(),
                    ],
                    limitations: vec![
                        "训练不稳定(VAE通病)".into(),
                        "隐含因子经济含义不直观".into(),
                        "计算开销较大".into(),
                    ],
                    reference: "FactorVAE (AAAI 2022, 2024 extensions)".into(),
                    reference_url: "https://aaai.org/papers/04468-factorvae/".into(),
                },
            ],
        },
        ModelCategory {
            name: "图神经网络 (GNN)".into(),
            description: "利用图结构建模资产间关系(行业链、资金流)，捕捉传统模型忽略的关系型因子".into(),
            models: vec![
                DlModelEntry {
                    id: "transgnn".into(),
                    name: "TransGNN".into(),
                    category: "GNN+Transformer".into(),
                    year: 2025,
                    architecture: "GNN-Transformer交替层 + 注意力采样".into(),
                    description: "融合GNN局部邻域特征和Transformer全局注意力，交替迭代层级。在推荐/选股任务Recall/NDCG提升20-30%。".into(),
                    key_innovation: "注意力采样模块选取最相关市场关系节点，多维度位置编码融入行业结构".into(),
                    input_data: vec!["价量数据".into(), "行业/产业链图".into(), "资金流向".into()],
                    output: "资产评分/排序".into(),
                    strengths: vec![
                        "Recall/NDCG提升20-30%".into(),
                        "捕捉行业轮动和产业链传导".into(),
                        "全局-局部信息耦合".into(),
                    ],
                    limitations: vec![
                        "图构建需要领域知识".into(),
                        "大规模图计算开销大".into(),
                        "动态图更新复杂".into(),
                    ],
                    reference: "TransGNN (2025)".into(),
                    reference_url: "https://hub.baai.ac.cn/view/32193".into(),
                },
                DlModelEntry {
                    id: "hist".into(),
                    name: "HIST".into(),
                    category: "GNN".into(),
                    year: 2024,
                    architecture: "Hierarchical Stock Transformer with Graph Attention".into(),
                    description: "层次化股票Transformer，将股票分为预定义概念(行业)和隐含概念两层，分别建模共享和个体因子。".into(),
                    key_innovation: "双层概念分解：预定义行业概念捕捉共性，隐含概念学习个股特异因子".into(),
                    input_data: vec!["个股特征序列".into(), "行业/概念标签".into()],
                    output: "个股收益率排序".into(),
                    strengths: vec![
                        "概念分解提升可解释性".into(),
                        "行业轮动建模自然".into(),
                        "A股实证表现优异(Qlib框架)".into(),
                    ],
                    limitations: vec![
                        "预定义概念需手动指定".into(),
                        "隐含概念数量需调参".into(),
                    ],
                    reference: "HIST (ACL 2022, extended 2024)".into(),
                    reference_url: "https://arxiv.org/abs/2110.13716".into(),
                },
                DlModelEntry {
                    id: "graph_mamba".into(),
                    name: "Graph-Mamba".into(),
                    category: "GNN+SSM".into(),
                    year: 2025,
                    architecture: "Graph Neural Network + Mamba State Space Model".into(),
                    description: "将Mamba(状态空间模型)与图结构结合，实现线性复杂度的长序列+图结构建模。".into(),
                    key_innovation: "用SSM替代Transformer的注意力，在图上实现O(n)复杂度的序列建模".into(),
                    input_data: vec!["时序价量数据".into(), "资产关系图".into()],
                    output: "节点级预测(个股收益)".into(),
                    strengths: vec![
                        "线性复杂度，可扩展到全市场".into(),
                        "长序列建模能力强".into(),
                        "图+序列双重建模".into(),
                    ],
                    limitations: vec![
                        "SSM对非线性模式捕捉有限".into(),
                        "金融领域验证尚少".into(),
                    ],
                    reference: "Graph-Mamba (2025 preprint)".into(),
                    reference_url: "https://arxiv.org/abs/2506.22084".into(),
                },
            ],
        },
        ModelCategory {
            name: "表格数据模型".into(),
            description: "针对结构化因子表格数据设计，兼顾深度学习表达力和可解释性".into(),
            models: vec![
                DlModelEntry {
                    id: "tabnet".into(),
                    name: "TabNet".into(),
                    category: "Attention-based Tabular".into(),
                    year: 2024,
                    architecture: "Sequential Attention + Feature Selection".into(),
                    description: "注意力机制驱动的表格数据深度学习模型，每步动态选择最重要的因子特征，兼具树模型的可解释性和深度学习的表达力。".into(),
                    key_innovation: "逐步注意力特征选择，每个决策步聚焦不同因子子集".into(),
                    input_data: vec!["结构化因子数据(财务/技术/另类)".into()],
                    output: "收益预测 + 特征重要性".into(),
                    strengths: vec![
                        "内置特征重要性(可解释)".into(),
                        "无需特征工程即可学习复杂模式".into(),
                        "端到端训练".into(),
                    ],
                    limitations: vec![
                        "小数据集表现不如梯度提升树".into(),
                        "训练超参数敏感".into(),
                    ],
                    reference: "TabNet (AAAI 2021, finance applications 2024)".into(),
                    reference_url: "https://arxiv.org/abs/1908.07442".into(),
                },
                DlModelEntry {
                    id: "alphanet".into(),
                    name: "AlphaNet".into(),
                    category: "Neural Factor".into(),
                    year: 2024,
                    architecture: "Deep Neural Network + Cross-sectional Feature Learning".into(),
                    description: "深度神经网络族，直接从价量基本面数据发现复杂的非线性alpha信号，具有更高的信息系数(IC)。".into(),
                    key_innovation: "多层非线性变换自动发现传统线性模型无法捕捉的alpha因子".into(),
                    input_data: vec!["价量数据".into(), "基本面数据".into(), "另类数据".into()],
                    output: "Alpha信号 + 收益预测".into(),
                    strengths: vec![
                        "高信息系数(IC)".into(),
                        "自动因子发现".into(),
                        "灵活适配多种数据源".into(),
                    ],
                    limitations: vec![
                        "黑箱模型，可解释性差".into(),
                        "过拟合风险".into(),
                        "需要大量历史数据训练".into(),
                    ],
                    reference: "AlphaNet family (various, 2020-2024)".into(),
                    reference_url: "https://github.com/nuglifeleoji/Factor-Research".into(),
                },
            ],
        },
        ModelCategory {
            name: "强化学习+注意力".into(),
            description: "将交易决策建模为序列决策问题，结合注意力机制进行多因子融合".into(),
            models: vec![
                DlModelEntry {
                    id: "dqn_bigru_attn".into(),
                    name: "DQN-BiGRU-Attention".into(),
                    category: "RL+Attention".into(),
                    year: 2024,
                    architecture: "DQN + Bi-GRU + Multi-Head ProbSparse Self-Attention".into(),
                    description: "深度Q学习+双向GRU+多头稀疏自注意力的混合架构，在多因子环境中学习交易动作。".into(),
                    key_innovation: "将多因子交互建模为注意力权重，强化学习直接优化交易收益".into(),
                    input_data: vec!["价量因子".into(), "基本面因子".into(), "情绪因子".into()],
                    output: "交易动作(买/卖/持有) + Q值".into(),
                    strengths: vec![
                        "直接优化交易收益(非预测收益)".into(),
                        "多因子自适应权重".into(),
                        "跨市场泛化能力强".into(),
                    ],
                    limitations: vec![
                        "训练不稳定(RL通病)".into(),
                        "样本效率低".into(),
                        "奖励函数设计困难".into(),
                    ],
                    reference: "Multi-factor stock trading via DQN with multi-head attention (2024)".into(),
                    reference_url: "https://link.springer.com/article/10.1007/s10489-024-05463-5".into(),
                },
            ],
        },
        ModelCategory {
            name: "多模态融合".into(),
            description: "融合文本(新闻/研报)、图像(K线图)、结构化数据的多模态因子模型".into(),
            models: vec![
                DlModelEntry {
                    id: "multimodal_gnn_transformer".into(),
                    name: "多模态GNN-Transformer".into(),
                    category: "Multimodal".into(),
                    year: 2025,
                    architecture: "Dynamic Graph + Temporal Transformer + Text Encoder".into(),
                    description: "结合动态图GNN与时间序列Transformer，支持同时输入基本面、新闻、产业链等异构数据，多时间尺度建模。".into(),
                    key_innovation: "异构数据统一编码到同一向量空间，动态图实时演化因子关系".into(),
                    input_data: vec!["基本面数据".into(), "新闻文本".into(), "产业链关系".into(), "价量序列".into()],
                    output: "多尺度收益预测 + 风险评估".into(),
                    strengths: vec![
                        "黑天鹅事件检测能力强".into(),
                        "行业切换感知".into(),
                        "多时间尺度预测".into(),
                        "可解释注意力热力图".into(),
                    ],
                    limitations: vec![
                        "系统复杂度极高".into(),
                        "数据对齐和清洗困难".into(),
                        "训练资源需求巨大".into(),
                    ],
                    reference: "GNN-Transformer集成金融分析系统 (2025)".into(),
                    reference_url: "https://pdf.hanspub.org/airr2025143_22610594.pdf".into(),
                },
            ],
        },
    ];

    ResearchKnowledgeBase {
        categories,
        collected: vec![],
        last_updated: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
    }
}

/// Generate LLM prompt for collecting latest DL factor model research.
pub fn build_collection_prompt(topic: &str) -> String {
    format!(
        r#"你是一位量化金融深度学习领域的资深研究员。请针对以下主题，搜集并整理最新的研究成果：

主题：{topic}

请按以下格式输出（JSON数组），每条研究包含：
- title: 研究标题
- summary: 100-200字核心要点总结（中文）
- source: 来源（论文/机构/平台）
- relevance: 与量化多因子策略的关联度评估（高/中/低）

重点关注：
1. 2024-2025年发表的最新论文和预印本
2. Transformer、GNN、VAE、强化学习在因子挖掘中的应用
3. 中国A股市场的实证研究
4. 可落地的工程化方案

请输出3-5条最有价值的研究成果。输出纯JSON数组，不要添加markdown格式或其他说明。"#,
        topic = topic
    )
}

/// Summary statistics for the knowledge base.
#[derive(Debug, Serialize)]
pub struct KnowledgeBaseSummary {
    pub total_models: usize,
    pub total_categories: usize,
    pub total_collected: usize,
    pub categories: Vec<CategorySummary>,
    pub last_updated: String,
}

#[derive(Debug, Serialize)]
pub struct CategorySummary {
    pub name: String,
    pub count: usize,
    pub description: String,
}

pub fn summarize_knowledge_base(kb: &ResearchKnowledgeBase) -> KnowledgeBaseSummary {
    let total_models: usize = kb.categories.iter().map(|c| c.models.len()).sum();
    KnowledgeBaseSummary {
        total_models,
        total_categories: kb.categories.len(),
        total_collected: kb.collected.len(),
        categories: kb
            .categories
            .iter()
            .map(|c| CategorySummary {
                name: c.name.clone(),
                count: c.models.len(),
                description: c.description.clone(),
            })
            .collect(),
        last_updated: kb.last_updated.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_base_has_all_categories() {
        let kb = build_knowledge_base();
        assert!(kb.categories.len() >= 6);
        let names: Vec<&str> = kb.categories.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"Transformer系列"));
        assert!(names.contains(&"VAE/生成式模型"));
        assert!(names.contains(&"图神经网络 (GNN)"));
        assert!(names.contains(&"表格数据模型"));
        assert!(names.contains(&"强化学习+注意力"));
        assert!(names.contains(&"多模态融合"));
    }

    #[test]
    fn test_knowledge_base_model_count() {
        let kb = build_knowledge_base();
        let total: usize = kb.categories.iter().map(|c| c.models.len()).sum();
        assert!(total >= 10, "Expected at least 10 models, got {}", total);
    }

    #[test]
    fn test_summary_matches() {
        let kb = build_knowledge_base();
        let summary = summarize_knowledge_base(&kb);
        assert_eq!(summary.total_categories, kb.categories.len());
        let total: usize = kb.categories.iter().map(|c| c.models.len()).sum();
        assert_eq!(summary.total_models, total);
    }

    #[test]
    fn test_collection_prompt_contains_topic() {
        let prompt = build_collection_prompt("Transformer因子挖掘");
        assert!(prompt.contains("Transformer因子挖掘"));
        assert!(prompt.contains("JSON"));
    }

    #[test]
    fn test_model_entries_have_required_fields() {
        let kb = build_knowledge_base();
        for cat in &kb.categories {
            for model in &cat.models {
                assert!(!model.id.is_empty());
                assert!(!model.name.is_empty());
                assert!(!model.description.is_empty());
                assert!(!model.strengths.is_empty());
                assert!(!model.limitations.is_empty());
                assert!(!model.reference_url.is_empty());
            }
        }
    }
}
