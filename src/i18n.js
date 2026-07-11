/* i18n.js — Internationalization (zh / en)
 * ------------------------------------------------------------
 * No DOM dependency except optional localStorage (guarded) and navigator.
 *
 * Usage:
 *   import { t, getLang, setLang, onLangChange } from '../i18n.js';
 *   t('brand.title');                 // current language
 *   t('status.shard', { done, total, file });
 *   setLang('en');                    // persists + notifies listeners
 * ------------------------------------------------------------ */

const STR = {
  // Brand
  'brand.title': {
    zh: 'LLM VRAM 计算器 - 零下载 GPU 显存估算工具',
    en: 'LLM VRAM Calculator - Zero-Download GPU Memory Estimator',
  },
  'brand.sub': {
    zh: '零下载解析 · 动态 VRAM 计算 · 双形态分发',
    en: 'Zero-download parsing · Dynamic VRAM estimation · Dual-form delivery',
  },

  // Controls
  'ctl.repoPlaceholder': {
    zh: 'org/repo，如 tencent/Hy3',
    en: 'org/repo, e.g. tencent/Hy3',
  },
  'ctl.analyze': { zh: '分析', en: 'Analyze' },
  'ctl.advanced': { zh: '高级（受限模型可选 Token）', en: 'Advanced (token for gated models)' },
  'ctl.tokenPlaceholder': { zh: 'hf_xxx（私有/受限仓库时填写）', en: 'hf_xxx (for private / gated repos)' },
  'ctl.batchSize': { zh: 'Batch Size：', en: 'Batch Size: ' },
  'ctl.contextLength': { zh: '上下文长度', en: 'Context Length' },

  // Overview
  'ov.title': { zh: '总览', en: 'Overview' },
  'ov.treeTitle': { zh: '张量名称树', en: 'Tensor name tree' },
  'ov.treeEmpty': { zh: '输入仓库 ID 并点击 Analyze 开始解析', en: 'Enter a repo ID and click Analyze to start parsing' },
  'ov.kvTitle': { zh: 'KV Cache 审计明细', en: 'KV Cache audit details' },
  'ctl.empty': { zh: '尚未分析', en: 'Not analyzed yet' },

  // Result summary card
  'sum.total': { zh: '总显存需求：', en: 'Total VRAM: ' },
  'sum.maxContext': { zh: '最大上下文：', en: 'Max context: ' },
  'sum.weights': { zh: '权重', en: 'Weights' },
  'sum.kv': { zh: 'KV', en: 'KV' },
  'sum.kvProfile': { zh: '模型架构档案：', en: 'Architecture Profile: ' },
  'sum.genericKV': { zh: 'KV 计算方式：', en: 'KV calculation: ' },
  'sum.kvLayout': { zh: 'KV 布局：', en: 'KV layout: ' },

  // Verified KV Cache audit output
  'kv.verified': { zh: '已验证', en: 'Verified' },
  'kv.approximate': { zh: '近似估算', en: 'Approximate' },
  'kv.warning': { zh: '未验证结果', en: 'Unverified result' },
  'kv.unsupported': { zh: 'KV Cache 无法验证', en: 'KV Cache could not be verified' },
  'kv.totalUnknown': { zh: 'KV Cache 未验证，因此总显存保持未知。', en: 'KV Cache is unverified, so total VRAM remains unknown.' },
  'kv.buffer': { zh: 'Buffer', en: 'Buffer' },
  'kv.layers': { zh: '层组', en: 'Layer group' },
  'kv.elements': { zh: '元素数', en: 'Elements' },
  'kv.dtype': { zh: 'DType', en: 'DType' },
  'kv.bytes': { zh: '字节 / GB', en: 'Bytes / GB' },
  'kv.formula': { zh: '公式', en: 'Formula' },
  'kv.repoId': { zh: '仓库：', en: 'Repository: ' },
  'kv.currentCommit': { zh: '当前 Commit：', en: 'Current commit: ' },
  'kv.auditedCommit': { zh: '计算依据的已审核 Commit：', en: 'Calculated from audited commit: ' },
  'kv.configDifferences': { zh: '配置差异：', en: 'Config differences: ' },
  'kv.issues': { zh: '无效输入：', en: 'Invalid inputs: ' },
  'kv.genericWarning': { zh: '通用 KV Cache 近似估算，不代表已验证布局。', en: 'Generic KV Cache approximation; this is not a Verified Layout.' },
  'kv.assumptions': { zh: '估算假设：', en: 'Assumptions: ' },
  'kv.assumption.uniform_full_context_attention': { zh: '所有层均采用完整上下文的统一注意力缓存', en: 'uniform full-context attention across all layers' },
  'kv.assumption.standard_key_value_buffers': { zh: '每层保存标准 key 与 value buffer', en: 'standard key and value buffers per layer' },
  'kv.assumption.kv_heads_defaulted_to_attention_heads': { zh: 'KV head 数量缺失，按 attention head 数量估算', en: 'KV head count defaulted to the attention head count' },
  'kv.assumption.head_dim_derived': { zh: 'head dimension 由 hidden size / attention head 数量推导', en: 'head dimension derived from hidden size / attention head count' },
  'kv.assumption.cache_dtype_defaulted': { zh: 'KV Cache DType 缺失或无法识别，默认使用', en: 'KV Cache DType missing or unknown; defaulted to' },
  'kv.assumption.cache_dtype_from_model_dtype': { zh: 'KV Cache DType 按模型 DType 推定为', en: 'KV Cache DType inferred from model DType as' },
  'kv.assumption.max_context_unavailable': { zh: 'config 未提供最大上下文限制', en: 'config does not provide a maximum context limit' },
  'kv.diag.unknown': { zh: '未知诊断', en: 'Unknown diagnostic' },
  'kv.diag.unsupported_model_architecture': { zh: '仓库未进入 Architecture Layout Catalog', en: 'Repository is not in the Architecture Layout Catalog' },
  'kv.diag.profile_input_out_of_range': { zh: 'batch 或 context 超出当前配置范围', en: 'Batch or context is outside the current config range' },
  'kv.diag.profile_calculation_out_of_range': { zh: 'KV 字节计算超出安全整数范围', en: 'KV byte calculation exceeds the safe integer range' },
  'kv.diag.invalid_model_provenance': { zh: '仓库来源或 Commit ID 无效', en: 'Repository provenance or commit ID is invalid' },
  'kv.diag.invalid_profile_config': { zh: '当前配置缺少算法输入或输入无效', en: 'Current config is missing required algorithm inputs or contains invalid values' },
  'kv.diag.generic_config_insufficient': { zh: '当前 config 不足以进行通用 KV Cache 估算', en: 'Current config is insufficient for a generic KV Cache estimate' },

  // Composition group titles
  'group.weight': { zh: '权重显存（稠密基础）', en: 'Weight VRAM (dense base)' },
  'group.moe': { zh: 'MoE 专家层', en: 'MoE expert layers' },
  'group.kv': { zh: 'KV Cache', en: 'KV Cache' },

  // Stats grid
  'stat.totalParams': { zh: '总参数', en: 'Total params' },
  'stat.layers': { zh: '层数', en: 'Layers' },
  'stat.moe': { zh: 'MoE', en: 'MoE' },
  'stat.moeYes': { zh: '是 × {n} 路由专家/层', en: 'Yes × {n} routed experts / layer' },
  'stat.no': { zh: '否', en: 'No' },
  'stat.arch': { zh: '架构', en: 'Arch' },
  'stat.shards': { zh: '分片数', en: 'Shards' },

  // Status line
  'status.enterRepo': { zh: '请输入仓库 ID（org/repo）', en: 'Please enter a repo ID (org/repo)' },
  'status.fetching': { zh: '解析中：拉取 config.json …', en: 'Parsing: fetching config.json …' },
  'status.shard': { zh: '解析分片头部 {done}/{total}：{file}', en: 'Parsing shard header {done}/{total}: {file}' },
  'status.done': { zh: '解析完成：{shards} 个分片，{tensors} 个张量', en: 'Done: {shards} shards, {tensors} tensors' },

  // Structure tree
  'tree.col.op': { zh: '张量路径', en: 'Tensor path' },
  'tree.col.shape': { zh: 'Shape', en: 'Shape' },
  'tree.col.dtype': { zh: 'DType', en: 'DType' },
  'tree.col.params': { zh: '参数量', en: 'Params' },
  'tree.col.vram': { zh: '显存', en: 'VRAM' },
  'tree.prefixHint': {
    zh: '每次展开一级；括号为实际子节点数，重复徽标为当前父级内的相同份数，悬停可查看具体 ID。',
    en: 'Expand one level at a time; parentheses show actual child counts, while repeat badges show matches within the current parent and reveal their IDs on hover.',
  },
  'tree.paramsUnit': { zh: '参数', en: 'params' },
  'tree.repeatIds': { zh: '{count} 份局部重复；IDs: {ids}', en: '{count} local matches; IDs: {ids}' },

  // Chart
  'chart.vramLabel': { zh: '显存占用 (GB)', en: 'VRAM usage (GB)' },
  'chart.gb': { zh: 'GB', en: 'GB' },

  // Composition category labels
  'cat.embedding': { zh: '嵌入层 Embedding', en: 'Embedding' },
  'cat.attn': { zh: '注意力层（稠密）', en: 'Attention (dense)' },
  'cat.mlp': { zh: 'MLP / FFN（稠密）', en: 'MLP / FFN (dense)' },
  'cat.norm': { zh: '归一化层（稠密）', en: 'Norm (dense)' },
  'cat.other': { zh: '其他（稠密）', en: 'Other (dense)' },
  'cat.lmhead': { zh: 'LM Head', en: 'LM Head' },
  'cat.expert': { zh: 'MoE 路由专家层', en: 'MoE routed expert layers' },
  'cat.weight': { zh: '权重（未细分）', en: 'Weights (unsplit)' },
  'cat.kv': { zh: 'KV Cache', en: 'KV Cache' },

  // Errors (user-facing)
  'err.noSafetensors': {
    zh: '该模型未提供 Safetensors 格式，无法进行远程 Range 碎片化解析，请更换现代大模型仓库。',
    en: 'This model does not provide the Safetensors format; remote Range-based header parsing is unavailable. Please use a modern model repo.',
  },
  'err.badRepoId': { zh: '请输入合法的仓库 ID，形如 org/repo', en: 'Please enter a valid repo ID, e.g. org/repo' },
  'err.configFetch': {
    zh: '无法获取 config.json（仓库是否存在或网络异常）：',
    en: 'Failed to fetch config.json (repo may not exist or network error): ',
  },
  'err.provenanceFetch': {
    zh: '无法解析仓库当前 Commit：',
    en: 'Failed to resolve the repository commit: ',
  },
  'err.badSafetensors': { zh: '文件 {file} 过短，不是合法的 safetensors', en: 'File {file} is too short to be a valid safetensors' },
  // Language toggle
  'lang.label': { zh: '语言', en: 'Language' },
};

const LS_KEY = 'hf-view-lang';
const listeners = new Set();

function detectLang() {
  try {
    if (typeof localStorage !== 'undefined') {
      const saved = localStorage.getItem(LS_KEY);
      if (saved === 'zh' || saved === 'en') return saved;
    }
  } catch {
    /* ignore */
  }
  try {
    if (typeof navigator !== 'undefined' && navigator.language) {
      return navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en';
    }
  } catch {
    /* ignore */
  }
  return 'en';
}

let current = detectLang();

export function getLang() {
  return current;
}

export function setLang(lang) {
  if (lang !== 'zh' && lang !== 'en') return;
  if (lang === current) return;
  current = lang;
  try {
    localStorage.setItem(LS_KEY, lang);
  } catch {
    /* ignore */
  }
  listeners.forEach((cb) => cb(lang));
}

export function onLangChange(cb) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function t(key, vars) {
  const entry = STR[key];
  let s = entry ? entry[current] ?? entry.en : key;
  if (vars && typeof vars === 'object') {
    for (const [k, v] of Object.entries(vars)) {
      s = s.replace(new RegExp('\\{' + k + '\\}', 'g'), v);
    }
  }
  return s;
}
