# 🚀 TIC 专家技能使用指南

## 📚 你的标准知识库架构

```
iCloud Drive
├── 法规和标准文件/
│   ├── ISO 标准 (7个)
│   ├── EU AI Act (5个版本)
│   ├── CEN/CENELEC 草案 (8个)
│   └── WP.29 法规 (目录)
├── AI法案协调标准/ (82个PDF) ⭐ 新增
│   ├── AI & Testing/ (15个) - AI测试标准
│   ├── Conformity Assessment/ (12个) - 合规评估
│   ├── Cybersecurity/ (7个) - 网络安全
│   ├── QMS/ (7个) - 质量管理
│   ├── ISO-IEC-5259/ (5个) - 数据质量
│   └── 其他主题分类 (14个子目录)
└── Standards/ (135个PDF) ⭐ 新增
    ├── 中文翻译标准/ (43个)
    ├── ISO 26262/ (16个)
    ├── GB/T 34590/ (12个)
    ├── ISO 3450X/ (7个)
    └── 其他通用标准 (57个)

↓ (通过索引映射)

~/.claude/skills/tic-expert/
├── SKILL.md                          # 主技能文件
├── references/                       # 知识结构
│   ├── iso-standards.md             # 70+ 标准详解
│   ├── eu-ai-act.md                 # EU AI Act 解读
│   ├── automotive-regs.md           # 汽车法规
│   └── audit-checklist.md           # 审核清单
└── standards/                        # 本地知识库
    ├── MASTER_FILE_INDEX.md         # 主文件索引 ⭐
    ├── MISSING_STANDBARDS_ANALYSIS.md # 缺失分析 ⭐
    └── KNOWLEDGE_BASE_STRUCTURE.md   # 目录结构
```

---

## 🎯 三种使用方式

### 方式 1：我直接读取你的原始文件（推荐）

**优点**：
- ✅ 不需要移动任何文件
- ✅ 保持 iCloud 同步
- ✅ 我可以访问最新的内容

**使用方法**：
```
你：帮我看一下 ISO 21448 第 6 章关于 SOTIF 分析的要求

我：[自动从 MASTER_FILE_INDEX.md 查找路径]
   [使用 Read 工具读取：/Users/sun1/.../ISO 21448, 2022-06-01 (1).pdf]
   [解读第 6 章内容，并结合 iso-standards.md 中的分析]
```

---

### 方式 2：基于知识结构回答（快速）

**优点**：
- ✅ 响应速度快
- ✅ 已经结构化的知识

**使用方法**：
```
你：ISO 21448 和 ISO 26262 的主要区别是什么？

我：[直接从 references/iso-standards.md 读取]
   [提供对比分析]
   [标注引用来源]
```

---

### 方式 3：使用完整文件映射表（推荐）⭐

**优点**：
- ✅ 涵盖所有217个标准文件
- ✅ 按目录结构组织
- ✅ 包含完整文件路径
- ✅ 智��识别标准类型

**使用方法**：
```
你：帮我找一下 ISO 24029 系列的所有文件

我：1. 查阅 NEW_FILES_MAPPING.md
   2. 找到以下文件：
      - AI法案协调标准/AI & Testing/ISOIEC 24029-2, 2023-08-01.pdf
      - AI法案协调标准/AI & Testing/ISOIEC TR 24029-1, 2021-03-01.pdf
   3. 提供完整路径和标准说明
```

**何时使用**：
- 需要查找特定标准的所有版本
- 需要浏览某个目录下的所有文件
- 需要了解完整的标准库结构

---

### 方式 4：组合模式（最强大）

**典型流程**：
```
你：我们的自动驾驶感知系统需要通过 EU AI Act 高风险 AI 认证，
    应该使用哪些标准进行测试？

我：1. [读取 eu-ai-act.md] 确认 Article 15 鲁棒性要求
    2. [读取 iso-standards.md] 找到对应的 ISO 标准
    3. [读取 MASTER_FILE_INDEX.md] 找到你已有的标准文件
    4. [如果需要，读取原始 PDF 文件]
    5. [提供综合建议]
```

---

## 💬 实际使用示例

### 示例 1：准备审核

```
你：下周要审核一个 L2+ 自动驾驶系统的 ISO 21448 符合性，
    我应该检查哪些文档？

我：1. 查找 ISO 21448 要求
   2. 提供审核检查清单（从 audit-checklist.md）
   3. 列出必备文档（SOTIF 分析、KNA/USS 识别等）
   4. 提供常见不符合项
```

### 示例 2：客户咨询

```
你：客户问他们的 AI 系统是否符合 EU AI Act，
    应该如何回答？

我：1. [使用四步分析法]
   2. 判断是否为高风险 AI
   3. 列出适用的 Article 条款
   4. 映射到 ISO 标准
   5. 提供合规建议
```

### 示例 3：查找标准条款

```
你：ISO 21448 中关于已知安全场景（KNA）的定义是什么？

我：1. [查找 MASTER_FILE_INDEX.md 中的 ISO 21448 路径]
   2. [读取原始 PDF 文件]
   3. [准确定位 KNA 定义]
   4. [提供上下文解释]
```

---

## 📂 如何添加新标准文件

### 步骤 1：将新文件放入 iCloud Drive

```
/Users/sun1/Library/Mobile Documents/com~apple~CloudDocs/Work/TUV/法规和标准文件/
```

建议文件命名：
- ISO 标准：`ISO 标准号 年份-月-日.pdf`
- CEN 标准：`CEN-CLC-JTC 21_编号_描述.pdf`

### 步骤 2：更新索引

编辑 `MASTER_FILE_INDEX.md`，添加新条目：

```markdown
| `ISO 26262-1, 2018-11-01.pdf` | automotive/iso-26262 | ISO 26262-1 词汇 | `/Users/sun1/.../ISO 26262-1, 2018-11-01.pdf` |
```

### 步骤 3：更新知识结构（可选）

如果需要，在 `references/iso-standards.md` 中添加该标准的详细解读。

---

## 🎓 学习路径建议

### 第 1 周：熟悉核心标准
1. **EU AI Act**：阅读中文版 + 英文版
2. **ISO/IEC 42001**：理解 AIMS 框架
3. **ISO 21448**：理解 SOTIF 概念

### 第 2 周：深入汽车行业
1. **ISO 26262**：功能安全基础
2. **ISO/SAE 21434**：网络安全
3. **WP.29 R155/R156**：法规要求

### 第 3 周：掌握测试方法
1. **ISO/IEC 29119-11**：AI 系统测试
2. **ISO/IEC 24029**：鲁棒性评估
3. **ETSI SAI**：安全测试

### 第 4 周：综合应用
1. 综合案例分析
2. 模拟审核
3. 客户咨询演练

---

## 📊 知识库维护

### 每月任务
- [ ] 检查 CEN/CENELEC 状态看板（N968）
- [ ] 更新新发布的 prEN 标准
- [ ] 审查 MASTER_FILE_INDEX.md 是否需要更新

### 每季度任务
- [ ] 评估是否需要购买新标准
- [ ] 更新 MISSING_STANDARDS_ANALYSIS.md
- [ ] 整理审核发现，更新 audit-checklist.md

### 每次审核后
- [ ] 记录新的不符合项模式
- [ ] 更新常见问题解答
- [ ] 补充实际案例

---

## 🆘 快速参考

### 常用命令

| 你想要... | 这样问 |
|----------|--------|
| 了解某个标准 | "给我讲解 ISO 21448 的核心要求" |
| 对比标准 | "ISO 21448 和 ISO 26262 的区别是什么？" |
| 准备审核 | "准备 ISO 42001 审核，我需要什么文档？" |
| 查找条款 | "ISO 21448 中关于 trigger event 的定义" |
| 客户建议 | "客户想通过 EU AI Act 认证，应该怎么做？" |
| 差距分析 | "我们的 AI 管理体系与 ISO 42001 有什么差距？" |

### 核心文件快速访问

| 需求 | 文件 |
|-----|------|
| **查找所有标准文件** | `NEW_FILES_MAPPING.md` ⭐ (217个文件的完整映射) |
| 快速索引 | `MASTER_FILE_INDEX.md` |
| 了解标准内容 | `references/iso-standards.md` |
| 检查缺失标准 | `MISSING_STANDARDS_ANALYSIS.md` |
| 准备审核 | `references/audit-checklist.md` |
| 查看 EU AI Act | `references/eu-ai-act.md` |

---

## 💡 最佳实践

### 1. 审核前准备
```
1. 使用 audit-checklist.md 生成检查清单
2. 从 iso-standards.md 查看具体要求
3. 如需详细信息，读取原始 PDF
4. 准备不符合项模板
```

### 2. 客户咨询
```
1. 理解客户场景（汽车行业？AI 系统类型？）
2. 使用四步分析法
3. 提供结构化建议
4. 给出实施路线图
```

### 3. 学习新标准
```
1. 先读 iso-standards.md 中的概述
2. 再读原始标准文件（如有）
3. 参考实际案例
4. 制作个人笔记
```

---

## 🎯 下一步行动

### 立即可做
1. ✅ 熟悉 `MASTER_FILE_INDEX.md`
2. ✅ 浏览 `MISSING_STANDARDS_ANALYSIS.md`
3. ✅ 选择一个核心标准深入学习

### 短期计划（1个月内）
1. ⭐ 下载免费的 NIST AI RMF 和 ETSI SAI
2. ⭐ 下载 EU 法规（GDPR, WP.29 R155/R156）
3. ⭐ 创建第一个标准的学习笔记

### 中期计划（3个月内）
1. ⭐ 采购 ISO 42001、ISO 26262、ISO 21434
2. ⭐ 建立 3 个核心标准的详细知识文件
3. ⭐ 完成一次模拟审核

---

**祝你成为 TIC 行业最顶尖的汽车 AI 合规专家！** 🎓

如有问题，随时询问。我会：
- 帮你查找标准文件
- 解读标准条款
- 准备审核材料
- 提供客户咨询建议
- 协助标准采购决策
