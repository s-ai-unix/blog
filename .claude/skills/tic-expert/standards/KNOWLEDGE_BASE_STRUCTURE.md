# TIC 专家标准知识库架构说明

## 📁 目录结构

```
~/.claude/skills/tic-expert/standards/
│
├── 00_INDEX.md                    # 主索引 - 快速查找
│
├── automotive/                    # 汽车行业专用标准
│   ├── iso-26262/                # 功能安全
│   │   ├── overview.md           # 标准概述
│   │   ├── part1-vocabulary.md   # 第1部分：词汇
│   │   ├── part3-concept.md      # 第3部分：概念阶段
│   │   ├── part4-system.md       # 第4部分：系统层面
│   │   ├── part5-hardware.md     # 第5部分：硬件层面
│   │   ├── part6-software.md     # 第6部分：软件层面
│   │   ├── asil-decision-tree.md # ASIL 决策树
│   │   ├── hara-template.md      # HARA 模板
│   │   └── audit-checklist.md    # 审核检查清单
│   │
│   ├── iso-21448-sotif/          # 预期功能安全
│   │   ├── overview.md
│   │   ├── trigger-events.md     # 触发事件识别
│   │   ├── kna-ussa-method.md    # 已知/未知安全场景方法
│   │   └── audit-checklist.md
│   │
│   ├── iso-21434-cybersecurity/  # 网络安全
│   │   ├── overview.md
│   │   ├── tara-method.md        # TARA 方法论
│   │   ├── threat-catalog.md     # 威胁目录
│   │   └── csms-checklist.md     # CSMS 审核清单
│   │
│   ├── iso-8800-ai-safety/       # AI 安全
│   │   ├── overview.md
│   │   ├── part1-concepts.md
│   │   ├── part2-design.md
│   │   ├── part3-testing.md
│   │   └── ai-specific-hazards.md # AI 特定危害
│   │
│   ├── iso-3450x-ads-scenarios/  # ADS 测试场景
│   │   ├── overview.md
│   │   ├── scenario-taxonomy.md  # 场景分类法
│   │   ├── odd-definition.md     # ODD 定义
│   │   └── scenario-library.md   # 场景库示例
│   │
│   └── ul-4600/                  # UL 4600
│       ├── overview.md
│       ├── safety-case-gsn.md    # Safety Case + GSN
│       └── audit-checklist.md
│
├── ai-management/                # 通用 AI 管理与技术标准
│   ├── iso-iec-42001-aims/       # AI 管理体系
│   │   ├── overview.md
│   │   ├── clause5-leadership.md
│   │   ├── clause6-planning.md
│   │   ├── clause7-support.md
│   │   ├── clause8-operation.md
│   │   ├── clause9-performance.md
│   │   ├── clause10-improvement.md
│   │   ├── pdca-cycle.md         # PDCA 循环
│   │   ├── integration-guide.md  # 与其他体系整合
│   │   └── audit-checklist.md
│   │
│   ├── iso-iec-23894-ai-risk/    # AI 风险管理
│   │   ├── overview.md
│   │   ├── risk-identification.md
│   │   ├── risk-analysis.md
│   │   ├── risk-evaluation.md
│   │   ├── risk-treatment.md
│   │   └── mapping-iso31000.md   # 与 ISO 31000 映射
│   │
│   ├── nist-ai-rmf/              # NIST AI RMF
│   │   ├── overview.md
│   │   ├── govern-function.md
│   │   ├── map-function.md
│   │   ├── measure-function.md
│   │   ├── manage-function.md
│   │   ├── playbook.md           # 实操指南
│   │   └── profile-template.md   # Profile 模板
│   │
│   ├── iso-iec-22989-terminology/ # AI 术语
│   │   ├── overview.md
│   │   ├── key-terms.md          # 核心术语表
│   │   └── classification.md     # AI 系统分类
│   │
│   ├── iso-iec-23053-ml-framework/ # ML 框架
│   │   ├── overview.md
│   │   ├── ml-lifecycle.md       # ML 生命周期
│   │   └── implementation-guide.md
│   │
│   └── iso-iec-tr-24028-trustworthiness/ # AI 可信度
│       ├── overview.md
│       ├── robustness.md
│       ├── explainability.md
│       ├── transparency.md
│       └── reproducibility.md
│
├── ai-quality/                   # AI 质量模型
│   └── iso-iec-25059-ai-quality/ # AI 质量模型
│       ├── overview.md
│       ├── quality-characteristics.md
│       ├── metrics.md            # 质量度量指标
│       └── assessment-method.md  # 评估方法
│
├── ai-testing/                   # AI 测试标准
│   ├── iso-iec-29119-11/         # AI 系统测试指南
│   │   ├── overview.md
│   │   ├── black-box-testing.md
│   │   ├── white-box-testing.md
│   │   ├── gray-box-testing.md
│   │   ├── metamorphic-testing.md # 变质测试
│   │   ├── adversarial-testing.md
│   │   ├── fairness-testing.md
│   │   └── test-strategy.md
│   │
│   ├── iso-iec-24029-robustness/ # 神经网络鲁棒性评估
│   │   ├── overview.md
│   │   ├── part1-overview.md     # Part 1: TR
│   │   ├── part2-formal-methods.md # Part 2: 形式化方法
│   │   ├── adversarial-robustness.md
│   │   ├── environmental-robustness.md
│   │   ├── ood-detection.md      # 分布外检测
│   │   ├── abstract-interpretation.md # 抽象解释
│   │   ├── smt-solving.md        # SMT 求解
│   │   ├── reachability-analysis.md # 可达性分析
│   │   └── verification-tools.md # 验证工具链
│   │
│   ├── iso-iec-ts-4213/          # ML 分类性能评估
│   │   ├── overview.md
│   │   ├── metrics-definition.md # 指标定义
│   │   ├── accuracy.md
│   │   ├── precision-recall.md
│   │   ├── f1-score.md
│   │   ├── roc-auc.md
│   │   ├── report-format.md      # 报告格式
│   │   └── statistical-tests.md  # 统计检验
│   │
│   ├── cen-elec-prEN/            # 欧盟协调标准（草案）
│   │   ├── overview.md
│   │   ├── conformity-assessment.md # 符合性评估
│   │   ├── robustness-safety-cybersecurity.md
│   │   ├── data-quality.md
│   │   ├── transparency.md
│   │   ├── human-oversight.md
│   │   └── roadmap.md            # 标准制定路线图
│   │
│   └── etsi-sai/                 # ETSI AI 安全测试
│       ├── overview.md
│       ├── gr-sai-001-threats.md # 威胁定义
│       ├── gr-sai-004-challenges.md # 技术挑战
│       ├── gr-sai-005-mitigation.md # 缓解策略
│       ├── adversarial-attacks.md # 对抗攻击
│       ├── data-poisoning.md     # 数据投毒
│       ├── model-inversion.md    # 模型反转
│       ├── model-extraction.md   # 模型窃取
│       ├── adversarial-training.md # 对抗训练
│       ├── detection-methods.md  # 检测方法
│       └── automotive-use-cases.md # 汽车应用案例
│
├── data-governance/              # 数据质量与治理
│   ├── iso-iec-5259-aml-data/    # AML 数据质量
│   │   ├── overview.md
│   │   ├── part1-quality-model.md
│   │   ├── part2-measures.md
│   │   ├── part3-processes.md
│   │   ├── part4-maturity.md
│   │   ├── part5-improvement.md
│   │   └── part6-indicators.md
│   │
│   ├── iso-8000-data-quality/    # 通用数据质量
│   │   ├── overview.md
│   │   ├── data-quality-principles.md
│   │   └── data-exchange.md
│   │
│   └── iso-iec-20547-big-data/   # 大数据架构
│       ├── overview.md
│       ├── reference-architecture.md
│       └── governance.md
│
├── risk-management/              # 风险管理基础
│   ├── iso-31000/                # 风险管理指南
│   │   ├── overview.md
│   │   ├── principles.md
│   │   ├── framework.md
│   │   └── process.md
│   │
│   └── iec-31010/                # 风险评估技术
│       ├── overview.md
│       ├── fmea.md               # FMEA 方法
│       ├── fta.md                # FTA 方法
│       ├── hazop.md              # HAZOP 方法
│       ├── stpa.md               # STPA 方法
│       ├── markov-analysis.md
│       └── monte-carlo.md
│
├── security/                     # 信息安全基础
│   ├── iso-iec-27001-isms/       # 信息安全管理体系
│   │   ├── overview.md
│   │   ├── annex-a-controls.md   # Annex A 控制措施详解
│   │   ├── soa-template.md       # 适用性声明模板
│   │   └── audit-checklist.md
│   │
│   └── iso-iec-27701-pims/       # 隐私信息管理
│       ├── overview.md
│       ├── pia-template.md       # 隐私影响评估模板
│       ├── gdpr-mapping.md       # GDPR 映射
│       └── data-subject-rights.md
│
├── ethics/                       # 伦理与社会影响
│   ├── iso-iec-tr-24368/         # 伦理关注点
│   │   ├── overview.md
│   │   ├── algorithmic-bias.md
│   │   ├── transparency.md
│   │   └── accountability.md
│   │
│   └── ieee-7000-series/         # IEEE 伦理设计
│       ├── overview.md
│       ├── ieee-7001-transparency.md
│       ├── ieee-7002-privacy.md
│       └── ieee-7003-bias.md
│
└── eu-regulations/               # 欧盟法规
    ├── eu-ai-act-2024/           # EU AI Act
    │   ├── overview.md
    │   ├── article6-high-risk.md # Article 6 高风险分类
    │   ├── article9-risk-mgmt.md
    │   ├── article10-data-gov.md
    │   ├── article11-tech-doc.md
    │   ├── article12-record-keeping.md
    │   ├── article13-transparency.md
    │   ├── article14-human-oversight.md
    │   ├── article15-accuracy.md
    │   ├── article43-conformity.md
    │   ├── annex-iii-high-risk.md
    │   └── compliance-checklist.md
    │
    ├── gdpr/                     # GDPR
    │   ├── overview.md
    │   ├── key-articles.md
    │   ├── data-subject-rights.md
    │   └── compliance-checklist.md
    │
    ├── eu-2018-858/              # 车辆型式认证
    │   ├── overview.md
    │   └── key-requirements.md
    │
    ├── eu-2019-2144/             # 通用安全法规
    │   ├── overview.md
    │   └── mandatory-systems.md  # 强制安全系统
    │
    └── wp29-regulations/         # WP.29 法规
        ├── r155-csms/            # CSMS
        │   ├── overview.md
        │   └── audit-checklist.md
        └── r156-sums/            # SUMS
            ├── overview.md
            └── audit-checklist.md
```

## 🎯 使用场景指南

### 场景 1: 准备 ISO 42001 审核计划
```
1. 打开 ai-management/iso-iec-42001-aims/overview.md
2. 查看 clause5-10 各条款的详细要求
3. 参考 audit-checklist.md 准备审核问题
4. 跨标准查阅 iso-iec-23894-ai-risk/ 了解风险管理细节
```

### 场景 2: 判断 AI 系统是否符合 EU AI Act
```
1. 打开 eu-regulations/eu-ai-act-2024/article6-high-risk.md
2. 查阅 annex-iii-high-risk.md 确认是否在清单中
3. 如确认为高风险，查阅 article9-15 各条款
4. 映射到对应 ISO 标准（如 iso-8800-ai-safety/）
```

### 场景 3: 审核中发现 ASIL D 系统的 SOTIF 问题
```
1. 打开 automotive/iso-26264/hara-template.md 查看安全目标
2. 打开 automotive/iso-21448-sotif/trigger-events.md
3. 参考 kna-ussa-method.md 分析客户是否遗漏场景
4. 撰写不符合报告
```

### 场景 4: 为客户提供 EU AI Act + ISO 42001 整合方案
```
1. 打开 eu-regulations/eu-ai-act-2024/overview.md
2. 打开 ai-management/iso-iec-42001-aims/integration-guide.md
3. 参考 nist-ai-rmf/playbook.md 提供实操建议
4. 准备整合路线图文档
```

### 场景 5: 评估 AI 模型的鲁棒性（ISO/IEC 24029）
```
1. 打开 ai-testing/iso-iec-24029-robustness/overview.md
2. 参考part1-overview.md了解评估框架
3. 如果需要形式化验证，查阅part2-formal-methods.md
4. 使用adversarial-robustness.md和ood-detection.md进行测试
5. 生成鲁棒性评估报告
```

### 场景 6: 对抗样本测试（ETSI SAI）
```
1. 打开 ai-testing/etsi-sai/gr-sai-001-threats.md了解威胁类型
2. 参考adversarial-attacks.md了解攻击方法
3. 使用gr-sai-005-mitigation.md设计防御措施
4. 实施对抗样本测试
5. 生成安全测试报告和缓解建议
```

### 场景 7: ML 分类模型性能评估（ISO/IEC TS 4213）
```
1. 打开 ai-testing/iso-iec-ts-4213/overview.md
2. 参考metrics-definition.md确认所需指标
3. 按照report-format.md准备性能报告
4. 使用statistical-tests.md进行统计显著性检验
5. 生成符合ISO标准的性能评估报告
```

### 场景 8: 准备 Notified Body 审核（基于prEN标准）
```
1. 打开 ai-testing/cen-elec-prEN/overview.md了解标准进展
2. 参考conformity-assessment.md准备审核流程
3. 按照robustness-safety-cybersecurity.md准备技术证据
4. 准备符合性评估技术文档
5. 预演Notified Body审核问题
```

### 场景 9: 高风险AI系统的全面测试包（EU AI Act合规）
```
1. EU AI Act Article 15 (鲁棒性)：
   - ai-testing/iso-iec-24029-robustness/
   - ai-testing/etsi-sai/adversarial-attacks.md
2. EU AI Act Article 10 (数据质量)：
   - data-governance/iso-iec-5259-aml-data/
3. EU AI Act Article 15 (准确性)：
   - ai-testing/iso-iec-ts-4213/
   - ai-quality/iso-iec-25059-ai-quality/
4. 生成综合测试报告，映射所有Article要求
```

### 场景 10: 为汽车客户提供AI安全测试服务
```
1. 识别适用标准：
   - automotive/iso-8800-ai-safety/ (汽车AI安全)
   - ai-testing/iso-iec-24029-robustness/ (鲁棒性)
   - ai-testing/etsi-sai/ (对抗攻击)
2. 设计测试策略：
   - 结合functional safety (ISO 26262)
   - 结合SOTIF (ISO 21448)
   - 结合cybersecurity (ISO 21434)
3. 执行测试并生成报告
4. 提供改进建议和认证路径
```

## 📝 每个标准文件应包含的内容

### overview.md
- 标准基本信息（编号、年份、名称）
- 适用范围
- 核心原则/要求摘要
- 与其他标准的关系
- 典型应用场景

### clause/section 详解文件
- 具体条款内容提取
- 条款解读（TIC 机构视角）
- 审核要点
- 常见不符合项
- 实施建议

### audit-checklist.md
- 分阶段的审核问题清单
- 必查文件清单
- 抽样建议
- 时间分配建议

### template 文件
- 客户可使用的模板（如 HARA、TARA、PIA）
- 填写说明
- 示例

## 🔄 维护策略

### 定期更新
- 每季度检查标准更新
- 标准修订时创建版本目录（如 v2018/, v2022/）
- 记录变更日志

### 知识沉淀
- 每次审核后更新"常见不符合项"
- 添加实际案例（脱敏后）
- 分享最佳实践

### 协作机制
- 标注"审核员自用"vs"客户可见"内容
- 建立审核员贡献机制
- 定期内部培训材料更新

## 🚀 快速开始

1. **从最常用的标准开始**：优先建立 ISO 26262、ISO 21448、ISO 42001、EU AI Act
2. **重质量轻数量**：每个标准的 overview 和 audit-checklist 优先
3. **边用边建**：实际审核中需要什么就补充什么
4. **复用现有结构**：参考 iso-standards.md 的内容，分解到具体文件

---

## 💡 建议

作为 TIC 机构的专家，你的知识库可以：
1. **成为机构资产**：团队共享，提升整体专业能力
2. **培训材料基础**：新审核员培训的教科书
3. **服务产品化**：打包成"合规工具包"销售给客户
4. **持续改进**：每次审核后更新，形成正向循环
