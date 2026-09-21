# 交付 1：配置、数据类、环境

范围：按主计划第6节串行完成第1项。关键逻辑使用英文注释；不实现模型、训练器、replay或训练状态机。

- [x] 核实计划、仓库规则、依赖与原生环境行为。
- [x] 实现严格配置、预算核算、公共数据类。
- [x] 实现数据manifest与Maze/FourRooms/同步向量环境。
- [x] 合成边界测试、真实数据reset/step/render验证。
- [x] 编写交付报告与后续接口说明。

使用planning-with-files持久记录。测试用标准库unittest，当前ctm环境没有pytest，避免修改旧测试依赖。

完成：24项测试通过；正式数据manifest生成成功；真实两环境各610transitions，均2次timeout/autoreset。停在交付1，未进入模型实现。
