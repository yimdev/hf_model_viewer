Type: prototype
Status: resolved
Blocked by: 01, 02, 03

## Question

什么样的 Architecture Profile 目录与每模型专用 resolver contract，能够把模型类标识、经审核的多字段 config/张量签名、证据版本和冲突原因显式关联起来，保证一次只选择一个专用布局，并在未知、歧义或版本漂移时 fail closed？

## Answer

目录与 dispatcher 已实现于 `src/vram/kv/catalog.js`、`src/vram/kv/index.js`：精确模型类标识只选候选。单一类标识下所有候选 Profile 分别验证全部登记签名，零个实际命中返回 signature mismatch、多个实际命中返回 conflict；不同已知类标识同时出现也直接返回 conflict。不存在通用回退。
