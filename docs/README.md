# 文档索引

本文档目录按用途分组。日常开发优先阅读“当前入口”和“运行维护”，历史规划类文档仅作背景参考。

## 当前入口

- [快速使用指南](quick_start_guide.md)：环境、运行、测试和常用命令。
- [架构说明](architecture.md)：应用层、核心包和数据流程。
- [重构进度](refactoring_progress_report.md)：当前重构状态、已完成事项和剩余风险。
- [测试实施状态](testing_implementation_report.md)：当前测试结果、覆盖率和下一步补测目标。

## 运行维护

- [配置管理](configuration_management.md)：`config/default.yaml`、环境变量和配置读取方式。
- [测试策略](testing_strategy.md)：测试分层、目标覆盖率和用例设计。
- [流程验证](workflow_verification.md)：训练、预测和回测流程验证记录。

## 研究和背景

- [种子分析](seed_analysis.md)：训练随机种子和历史复现实验说明。
- [事件状态模型指南](event_regime_model_guide.md)：事件模型相关说明。
- [架构评估](architecture_review.md)：早期架构问题和改进建议。
- [重构计划](refactoring_plan.md)：早期重构路线图，部分内容已完成，当前状态以 [重构进度](refactoring_progress_report.md) 为准。

## 过时文档处理原则

- 根目录只保留 `README.md` 作为项目入口。
- 阶段性实施报告统一合并到 `docs/refactoring_progress_report.md` 和 `docs/testing_implementation_report.md`。
- 覆盖率目录、pytest 缓存、`__pycache__` 等运行产物不保留。
