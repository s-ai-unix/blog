# 标准文件缺失分析报告

> **生成时间**：2025-01-23
> **对比基准**：tic-expert skill 中定义的 70+ 标准/法规/技术规范
> **现有文件位置**：
> - `/Users/sun1/Library/Mobile Documents/com~apple~CloudDocs/Work/TUV/法规和标准文件`
> - `/Users/sun1/Library/Mobile Documents/com~apple~CloudDocs/Work/TUV/AI法案协调标准`

---

## ✅ 已成功映射的标准（21个）

### 📗 ISO/IEC 标准
| 标准号 | 标准名称 | 文件名 | 状态 |
|-------|---------|-------|------|
| ISO 21448:2022 | SOTIF 预期功能安全 | `ISO 21448, 2022-06-01 (1).pdf` | ✅ 已映射 |
| ISO/PAS 8800:2024 | AI 安全 | `ISO_PAS+8800-2024_中文版.pdf` | ✅ 已映射 |
| ISO/IEC 22989:2022 | AI 术语 | `ISOIEC 22989, 2022-07-01.pdf` | ✅ 已映射 |
| ISO/IEC 23894:2023 | AI 风险管理 | `ISOIEC 23894, 2023-02-01.pdf` | ✅ 已映射 |
| ISO/IEC DIS 27090 | AI 安全和网络安全 | `ISO IEC DIS 27090-2025.pdf` | ✅ 已映射 |
| ISO/IEC TR 24028:2020 | AI 可信度 | `ISO IEC TR 24028 2020.pdf` | ✅ 已映射 |
| ISO/IEC DIS 42006 | AIMS 指南 | `ISOIEC DIS 42006, 2023-10-01.pdf` | ✅ 已映射 |
| ISO/IEC TR 5469:2024 | AI 审计 | `ISO-IEC-TR-5469-2024-01-01.pdf` | ✅ 已映射 |

### 🇪🇺 欧盟法规
| 法规 | 描述 | 文件名 | 状态 |
|-----|------|-------|------|
| EU AI Act 2024 | 人工智能法案 | ��个版本（中文/英文） | ✅ 已映射 |
| Machinery Regulation | 机械法规 | `Machinery_CELEX_32023R1230_EN_TXT.pdf` | ✅ 已映射 |
| MDR | 医疗器械法规 | `MDR_CELEX_32017R0745_EN_TXT.pdf` | ✅ 已映射 |

### 📋 CEN/CENELEC 协调标准（草案）
| 文档号 | 描述 | 状态 |
|-------|------|------|
| CEN-CLC-JTC 21_N831 | AI Trustworthiness Framework | ✅ 已映射 |
| CEN-CLC-JTC 21_N946 | AI Risk Management | ✅ 已映射 |
| CEN-CLC-JTC 21_N995 | Architecture of standards | ✅ 已映射 |
| CEN-CLC-JTC 21 N773 | Status Dashboard 2024 | ✅ 已映射 |
| CEN-CLC-JTC 21 N968 | Status Dashboard 2025 | ✅ 已映射 |
| CEN-CLC-JTC 21-WG 2_N1000 | QMS Annex X | ✅ 已映射 |
| CEN-CLC-JTC 21-WG 2_N1037 | QMS WD21 | ✅ 已映射 |
| CEN-CLC_FGR | RoadMap AI | ✅ 已映射 |

### 🚗 WP.29 法规
| 法规 | 描述 | 状态 |
|-----|------|------|
| UN Regulation on vehicles | 车辆型式认证 | ✅ 已映射（目录） |
| WP.29 系列 | 其他 WP.29 法规 | ✅ 已映射（目录） |

---

## ❌ 缺失的标准文件（按优先级排序）

### 🔴 高优先级（核心汽车 AI 标准）

#### 1. 汽车行业专用标准（6个缺失）

| 标准号 | 标准名称 | 分类 | 优先级 | 说明 |
|-------|---------|------|--------|------|
| **ISO 26262:2018** | Road vehicles - Functional safety | automotive/iso-26262 | ⭐⭐⭐⭐⭐ | 功能安全圣经，ASIL 评级必备 |
| **ISO/SAE 21434:2021** | Road vehicles - Cybersecurity | automotive/iso-21434-cybersecurity | ⭐⭐⭐⭐⭐ | 网络安全，CSMS 审核 |
| **ISO 34501** | ADS Test Scenarios - Vocabulary | automotive/iso-3450x-ads-scenarios | ⭐⭐⭐⭐ | 自动驾驶测试场景 |
| **ISO 34502** | ADS Test Scenarios - Generation | automotive/iso-3450x-ads-scenarios | ⭐⭐⭐⭐ | 场景生成方法 |
| **ISO 34503** | ODD Classification | automotive/iso-3450x-ads-scenarios | ⭐⭐⭐⭐ | ODD 定义和分类 |
| **UL 4600** | Safety of Autonomous Products | automotive/ul-4600 | ⭐⭐⭐⭐ | Safety Case 方法论 |

**获取建议**：
- ISO 26262 和 ISO/SAE 21434 是汽车行业必备，建议立即购买
- ISO 3450x 系列适合自动驾驶团队
- UL 4600 可作为参考（北美市场）

---

### 🟡 中优先级（通用 AI 管理与测试）

#### 2. 通用 AI 管理标准（5个缺失）

| 标准号 | 标准名称 | 分类 | 优先级 | 说明 |
|-------|---------|------|--------|------|
| **ISO/IEC 42001:2023** | AI Management System (AIMS) | ai-management/iso-iec-42001-aims | ⭐⭐⭐⭐⭐ | AI 管理体系核心，EU AI Act 合规关键 |
| **NIST AI RMF 1.0** | AI Risk Management Framework | ai-management/nist-ai-rmf | ⭐⭐⭐⭐ | 全球最流行的 AI 风险框架 |
| **ISO/IEC 23053:2022** | Framework for AI Systems Using ML | ai-management/iso-iec-23053-ml-framework | ⭐⭐⭐ | ML 生命周期框架 |
| **ISO/IEC 25059:2023** | Quality model for AI systems | ai-quality/iso-iec-25059-ai-quality | ⭐⭐⭐⭐ | AI 质量模型 |
| **ISO/IEC 29119-11:2020** | Testing of AI-based systems | ai-testing/iso-iec-29119-11 | ⭐⭐⭐⭐ | AI 系统测试指南 |

**获取建议**：
- ISO/IEC 42001 是 2024 年最热门标准，必须购买
- NIST AI RMF 是**免费公开**的，可直接下载
- ISO/IEC 23053、25059、29119-11 可作为第二梯队购买

---

#### 3. AI 测试与鲁棒性（4个缺失）

| 标准号 | 标准名称 | 分类 | 优先级 | 说明 |
|-------|---------|------|--------|------|
| **ISO/IEC 24029-1** | Neural Network Robustness - Overview | ai-testing/iso-iec-24029-robustness | ⭐⭐⭐⭐ | 鲁棒性评估框架 |
| **ISO/IEC 24029-2** | Neural Network Robustness - Formal Methods | ai-testing/iso-iec-24029-robustness | ⭐⭐⭐⭐ | 形式化验证 |
| **ISO/IEC TS 4213:2022** | ML Classification Performance | ai-testing/iso-iec-ts-4213 | ⭐⭐⭐ | 分类模型性能评估 |
| **ETSI GR SAI 001** | Security Problem Definition | ai-testing/etsi-sai | ⭐⭐⭐⭐ | AI 安全威胁定义 |
| **ETSI GR SAI 004** | Problem Statement | ai-testing/etsi-sai | ⭐⭐⭐ | 安全挑战 |
| **ETSI GR SAI 005** | Mitigation Strategy | ai-testing/etsi-sai | ⭐⭐⭐⭐ | 缓解策略 |

**获取建议**：
- ISO/IEC 24029 系列适合提供深度测试服务
- ETSI GR SAI 系列是**免费公开**的，可直接下载

---

### 🟢 低优先级（基础标准与支持工具）

#### 4. 数据质量与治理（3个缺失）

| 标准号 | 标准名称 | 分类 | 优先级 | 说明 |
|-------|---------|------|--------|------|
| **ISO/IEC 5259 Series** | Data Quality for AML (6 parts) | data-governance/iso-iec-5259-aml-data | ⭐⭐⭐ | AI/ML 数据质量 |
| **ISO 8000 Series** | Data Quality | data-governance/iso-8000-data-quality | ⭐⭐ | 通用数据质量 |
| **ISO/IEC 20547 Series** | Big Data Reference Architecture | data-governance/iso-iec-20547-big-data | ⭐⭐ | 大数据架构 |

**获取建议**：
- 如果提供数据治理服务，ISO/IEC 5259 值得购买
- 其他两个可作为参考

---

#### 5. 风险管理与信息安全（4个缺失）

| 标准号 | 标准名称 | 分类 | 优先级 | 说明 |
|-------|---------|------|--------|------|
| **ISO 31000:2018** | Risk Management Guidelines | risk-management/iso-31000 | ⭐⭐⭐ | 风险管理基础 |
| **IEC 31010:2019** | Risk Assessment Techniques | risk-management/iec-31010 | ⭐⭐⭐ | FMEA, FTA, HAZOP, STPA |
| **ISO/IEC 27001:2022** | Information Security Management | security/iso-iec-27001-isms | ⭐⭐⭐ | ISMS 信息安全 |
| **ISO/IEC 27701:2019** | Privacy Information Management | security/iso-iec-27701-pims | ⭐⭐⭐ | PIMS 隐私保护 |

**获取建议**：
- 这些是基础标准，如果公司已有可复用
- IEC 31010 包含 FMEA、FTA 等工具，对审核很有帮助

---

#### 6. 伦理与社会影响（2个缺失）

| 标准号 | 标准名称 | 分类 | 优先级 | 说明 |
|-------|---------|------|--------|------|
| **ISO/IEC TR 24368:2022** | Ethical and Societal Concerns | ethics/iso-iec-tr-24368 | ⭐⭐ | AI 伦理关注点 |
| **IEEE 7000 Series** | Ethically Aligned Design | ethics/ieee-7000-series | ⭐⭐ | 伦理设计 |

**获取建议**：
- 可作为高级服务提供
- 不是当前审核刚需

---

#### 7. 其他欧盟法规（5个缺失）

| 法规 | 描述 | 分类 | 优先级 |
|-----|------|------|--------|
| **GDPR** | 通用数据保护条例 | eu-regulations/gdpr | ⭐⭐⭐⭐ |
| **EU 2018/858** | 车辆型式认证 | eu-regulations/eu-2018-858 | ⭐⭐⭐⭐ |
| **EU 2019/2144** | 通用安全法规 | eu-regulations/eu-2019-2144 | ⭐⭐⭐⭐ |
| **WP.29 R155** | CSMS 网络安全 | eu-regulations/wp29-regulations/r155-csms | ⭐⭐⭐⭐⭐ |
| **WP.29 R156** | SUMS 软件更新 | eu-regulations/wp29-regulations/r156-sums | ⭐⭐⭐⭐⭐ |

**获取建议**：
- 这些法规都是**免费公开**的，可从欧盟官网下载
- WP.29 R155/R156 是汽车行业必读

---

## 📊 缺失统计

### 按分类统计

| 分类 | 已有 | 缺失 | 完整率 |
|-----|------|------|--------|
| 汽车行业专用 | 2 | 6 | 25% |
| 通用 AI 管理 | 6 | 5 | 55% |
| AI 测试 | 1 | 6 | 14% |
| AI 质量模型 | 0 | 1 | 0% |
| 数据治理 | 0 | 3 | 0% |
| 风险管理 | 0 | 4 | 0% |
| 信息安全 | 0 | 4 | 0% |
| 伦理 | 0 | 2 | 0% |
| 欧盟法规 | 3 | 5 | 38% |
| CEN/CENELEC 草案 | 8 | 若干 | 70% |
| **总计** | **20** | **40+** | **33%** |

---

## 🎯 推荐采购计划

### 阶段 1：立即购买（2025 Q1）

| 标准号 | 预估费用 | 紧急度 | 理由 |
|-------|---------|--------|------|
| **ISO/IEC 42001:2023** | ~€200 | 🔴 极高 | EU AI Act 合规核心 |
| **ISO 26262:2018** (全套) | ~€2000 | 🔴 极高 | 汽车功能安全必备 |
| **ISO/SAE 21434:2021** | ~€200 | 🔴 极高 | 汽车网络安全必备 |
| **ISO/IEC 25059:2023** | ~€150 | 🟠 高 | AI 质量模型 |
| **ISO/IEC 29119-11:2020** | ~€150 | 🟠 高 | AI 测试指南 |
| **ISO/IEC 24029-1 & 2** | ~€300 | 🟠 高 | 鲁棒性评估 |
| **WP.29 R155 & R156** | 免费 | 🔴 极高 | 法规要求 |

**小计**：~€3000 + 免费法规

---

### 阶段 2：2025 Q2-Q3

| 标准号 | 预估费用 | 理由 |
|-------|---------|------|
| **ISO 3450x Series** | ~€600 | 自动驾驶测试 |
| **UL 4600** | ~€300 | Safety Case |
| **ISO/IEC 23053:2022** | ~€150 | ML 框架 |
| **ISO/IEC 5259 Series** | ~€400 | 数据质量 |
| **IEC 31010:2019** | ~€200 | 风险评估工具 |

**小计**：~€1650

---

### 阶段 3：免费资源（立即下载）

| 标准号 | 来源 | 链接 |
|-------|------|------|
| **NIST AI RMF 1.0** | NIST 官网 | https://www.nist.gov/itl/ai-risk-management-framework |
| **ETSI GR SAI 001/004/005** | ETSI 官网 | https://www.etsi.org/committee/1414-sai |
| **GDPR** | EUR-Lex | https://eur-lex.europa.eu/ |
| **EU 2018/858** | EUR-Lex | https://eur-lex.europa.eu/ |
| **EU 2019/2144** | EUR-Lex | https://eur-lex.europa.eu/ |
| **WP.29 R155** | UNECE | https://unece.org/ |
| **WP.29 R156** | UNECE | https://unece.org/ |

**小计**：€0

---

## 💡 替代方案

如果预算有限，可以考虑：

### 1. 使用公开资源
- ISO 官方网站的白皮书和摘要
- 学术论文中的标准引用和解读
- 行业协会的公开资料

### 2. 与其他机构合作
- 与有这些标准的机构建立合作关系
- 参加标准制定的公开会议
- 加入 CEN/CENELEC 或 ISO 的工作组

### 3. 分阶段采购
- 先买最核心的（ISO 42001, ISO 26262, ISO 21434）
- 根据客户需求再采购其他标准
- 关注 ISO 在线订阅服务（可能更经济）

---

## 🔄 与现有文件的互补

你现有的文件已经覆盖了：
- ✅ EU AI Act（多版本，中英文）
- ✅ CEN/CENELEC 草案（非常新的工作文档）
- ✅ AI Trustworthiness 框架
- ✅ 部分核心 ISO 标准

**优势**：
- 你的 CEN/CENELEC 文件非常新（2025年），可以直接跟踪标准制定进展
- 有 EU AI Act 的中文翻译，便于理解
- 有各种培训材料和演示文稿

**建议**：
保持现有文件结构，通过索引系统访问，缺什么补什么。

---

**最后更新**：2025-01-23
**下次审查**：2025-06-01 或当有新标准发布时
