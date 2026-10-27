# 验证记录

2026-10-23：Windows / Python 环境中执行。

```powershell
python -X utf8 -m unittest discover -s Baseline/tests -v
python Baseline/run.py demo --output Baseline/runs/demo-verified
```

测试覆盖：原失败采样一致性，SearchQA 答案抽取与 EM/F1，HotpotQA yes/no F1，gold/supporting-facts 不进入正常解题 prompt，原上下文截断规则，句子位置保留，划分/ID 校验，数据冻结，训练记忆隔离与去重，成功配对，有界 JSON 反馈，预算调用前预留，失败请求保守计费，reasoning 不重复记账，原变异修改范围，修复/退化配对比较，两个数据集 N/I/B/C 全流程，冻结后技能篡改拒绝。

本机验证使用已有 Python、PyYAML、PyArrow；正式 TokenCounter 依赖 transformers 和用户固定的本地 tokenizer。**尚未验证真实模型服务、A6000 性能或真实 EM/F1。** 合成 mock 运行仅验证程序逻辑，不计为 benchmark 成绩。

真实数据仅验证下载版本、SHA256、ID 划分和字段完整性，没有运行模型查看测试成绩。HotpotQA 公共数据的 test 缺失明确保留，不能据此声称完成课程版测试。

最终的运行结果与 Python/依赖版本保存在 `validation_report.json`。
