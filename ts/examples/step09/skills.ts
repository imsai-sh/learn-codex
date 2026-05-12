// =============================================================================
// step05+ / skills.ts —— Skill 加载器（step05 ~ step09 共用）
// -----------------------------------------------------------------------------
// Skill 是一个本地的“可按需加载的指令片段”。约定：
//   - 每个 skill 是一个目录，目录里有一个 SKILL.md；
//   - SKILL.md 顶部可以放 YAML frontmatter（name / description）；
//   - 用户在输入里用 $skill-name 显式提到时，它对应的 SKILL.md 会被注入到本轮上下文。
//
// 这一组函数负责：
//   1. load_skills:       扫描目录，抽出元数据（name / description / 完整路径）；
//   2. render_skills_section: 把所有 skill 的目录写进 system prompt，告诉模型
//      “你拥有这些 skill，但只有用户显式提到才生效”；
//   3. collect_explicit_skill_mentions: 从用户输入里识别 $skill-name；
//   4. build_skill_injection_messages: 真正把命中的 SKILL.md 内容读出来包成消息。
// =============================================================================

import { promises as fs } from "node:fs";
import * as path from "node:path";

/** 单个 skill 的元数据。 */
export interface SkillMetadata {
  name: string;
  description: string;
  path: string;
}

/** 扫描一次目录得到的结果（含潜在的告警，便于上层打印）。 */
export interface SkillLoadOutcome {
  skills: SkillMetadata[];
  warnings: string[];
}

/**
 * 递归扫描 root 下所有 SKILL.md，按 name 排序返回。
 * 任何一条加载失败只会写入 warnings，不会抛出异常 —— 这与 Rust 版语义一致。
 */
export async function loadSkills(root: string): Promise<SkillLoadOutcome> {
  const outcome: SkillLoadOutcome = { skills: [], warnings: [] };
  await discoverSkills(root, outcome);
  outcome.skills.sort((a, b) => a.name.localeCompare(b.name));
  return outcome;
}

async function discoverSkills(root: string, outcome: SkillLoadOutcome): Promise<void> {
  let entries: import("node:fs").Dirent[];
  try {
    entries = await fs.readdir(root, { withFileTypes: true });
  } catch {
    // 目录不存在直接静默 —— 与 Rust `let Ok(entries) = ...` 行为一致。
    return;
  }

  for (const entry of entries) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) {
      await discoverSkills(full, outcome);
      continue;
    }
    if (entry.name !== "SKILL.md") continue;
    try {
      const skill = await parseSkillFile(full);
      outcome.skills.push(skill);
    } catch (e) {
      outcome.warnings.push(`failed to parse ${full}: ${(e as Error).message}`);
    }
  }
}

async function parseSkillFile(filePath: string): Promise<SkillMetadata> {
  const contents = await fs.readFile(filePath, "utf8");
  const frontmatter = extractFrontmatter(contents);
  // 没显式给 name 时，回退到上层目录名（"foo/SKILL.md" → "foo"）。
  const fallbackName = path.basename(path.dirname(filePath)) || "skill";
  const name = (frontmatter && frontmatterValue(frontmatter, "name")) ?? fallbackName;
  const description =
    (frontmatter && frontmatterValue(frontmatter, "description")) ?? "No description provided.";
  return { name, description, path: filePath };
}

/** 抽取 markdown 顶部 `---` ... `---` 之间的内容；不存在则返回 null。 */
function extractFrontmatter(contents: string): string | null {
  const lines = contents.split(/\r?\n/);
  if (lines[0] !== "---") return null;
  const fmLines: string[] = [];
  for (let i = 1; i < lines.length; i++) {
    if (lines[i] === "---") return fmLines.join("\n");
    fmLines.push(lines[i]);
  }
  return null;
}

/** 在 frontmatter 文本里查 key: value。 */
function frontmatterValue(frontmatter: string, key: string): string | undefined {
  for (const line of frontmatter.split(/\r?\n/)) {
    const idx = line.indexOf(":");
    if (idx < 0) continue;
    const lhs = line.slice(0, idx).trim();
    if (lhs !== key) continue;
    return line.slice(idx + 1).trim().replace(/^"+|"+$/g, "");
  }
  return undefined;
}

/**
 * 把所有 skill 渲染成一段“目录 + 用法”的 markdown，作为 system prompt 的附加段落。
 * 没有任何 skill 时返回 null，调用方可以选择跳过拼接。
 */
export function renderSkillsSection(skills: SkillMetadata[]): string | null {
  if (skills.length === 0) return null;

  const lines: string[] = [
    "## Skills",
    "A skill is a local instruction file stored as `SKILL.md`.",
    "### Available skills",
  ];
  for (const s of skills) {
    lines.push(`- ${s.name}: ${s.description} (file: ${s.path})`);
  }
  lines.push("### How to use skills");
  lines.push(
    "- If the user explicitly mentions a skill with `$skill-name`, you should use that skill for the current turn.",
  );
  lines.push(
    "- When a skill is selected, read and follow its `SKILL.md` instructions before taking action.",
  );
  lines.push(
    "- Do not assume a skill is active unless the user explicitly mentions it for the turn.",
  );
  return lines.join("\n");
}

/**
 * 在用户输入里找 $skill-name，再和已加载的 skill 列表求交集。
 * 这一步只能识别“用户明确点名”的 skill，不会自动触发。
 */
export function collectExplicitSkillMentions(
  input: string,
  skills: SkillMetadata[],
): SkillMetadata[] {
  const mentioned = new Set(extractSkillMentions(input));
  return skills.filter((s) => mentioned.has(s.name));
}

/** 把命中的 SKILL.md 真正读出来，包装成可注入到 system 段的字符串。 */
export async function buildSkillInjectionMessages(
  skills: SkillMetadata[],
): Promise<{ messages: string[]; warnings: string[] }> {
  const messages: string[] = [];
  const warnings: string[] = [];
  for (const skill of skills) {
    try {
      const contents = await fs.readFile(skill.path, "utf8");
      messages.push(
        `<skill>\n<name>${skill.name}</name>\n<path>${skill.path}</path>\n${contents}\n</skill>`,
      );
    } catch (e) {
      warnings.push(`failed to load skill ${skill.name} at ${skill.path}: ${(e as Error).message}`);
    }
  }
  return { messages, warnings };
}

/**
 * 从一段任意文本中抽取所有 $skill-name token。
 * 字符集允许 [a-zA-Z0-9_:-]，与 Rust 实现保持一致。
 */
function extractSkillMentions(text: string): string[] {
  const out: string[] = [];
  let i = 0;
  while (i < text.length) {
    if (text[i] !== "$") {
      i++;
      continue;
    }
    let j = i + 1;
    while (j < text.length) {
      const c = text[j];
      // 仅 ASCII 字母数字 / - / _ / :
      if (/[A-Za-z0-9_\-:]/.test(c)) j++;
      else break;
    }
    if (j > i + 1) out.push(text.slice(i + 1, j));
    i = j;
  }
  return out;
}
