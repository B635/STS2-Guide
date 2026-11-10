// The terminal side of `coach ask`: a readline chat, or a fixed list of
// questions for scripts. The answer is streamed while the lead writes it,
// then the checks run; if they change it, the checked version follows.
// Tool activity shows as short grey lines.

import { createInterface } from "node:readline";
import { ask, shownAnswer } from "./ask.js";
import type { Answer, AskContext, AskEvent, Session } from "./ask.js";

const GREY = "\x1b[2m", RESET = "\x1b[0m";

interface Out {
  write(text: string): void;
}

/** Turns the lead's events into terminal lines. `progress` gets the grey lines, `answer` the text. */
export function printer(progress: Out, answer: Out, color: boolean) {
  const grey = (s: string) => (color ? `${GREY}${s}${RESET}` : s);
  let streaming: "draft" | "rewrite" | null = null;
  const endStream = () => {
    if (streaming) answer.write("\n");
    streaming = null;
  };
  const line = (s: string) => {
    endStream();
    progress.write(grey(s) + "\n");
  };
  return {
    event(e: AskEvent): void {
      switch (e.type) {
        case "plan":
          return line(`计划：${e.text}`);
        case "tool":
          return line(`  · 查询 ${e.tool}${e.detail ? ` ${e.detail}` : ""}`);
        case "investigate":
          return line(`  · 调查 ${e.decisions.join("、")}`);
        case "investigated":
          return line(`    ${e.decision}：${e.status === "verified" ? "结论已核对" : e.status === "cached" ? "沿用前面的调查" : "结论没通过核对，只用引擎的比较"}`);
        case "draft":
        case "rewrite":
          if (streaming !== e.type) {
            endStream();
            answer.write(e.type === "draft" ? "\n回答（核对中）：\n" : "\n重写（核对中）：\n");
            streaming = e.type;
          }
          return answer.write(e.delta);
        case "draft_discarded":
          return line("  （继续查证）");
        case "check":
          return line(e.failed.length ? `  · 程序核对：${e.failed.length} 句的数字或名称在工具结果里找不到` : "  · 程序核对：通过");
        case "critic":
          return line(e.issues.length ? `  · 审查：${e.issues.length} 处说法没有依据` : "  · 审查：通过");
      }
    },
    /** After the checks: say it passed, or print the checked version. */
    done(a: Answer, streamed: boolean): void {
      endStream();
      if (streamed && a.status === "passed") {
        progress.write(grey("[已核对：数字、名称和每个说法都能对上引擎的结果]") + "\n\n");
        return;
      }
      if (streamed) answer.write("\n核对后的回答：\n");
      answer.write(shownAnswer(a) + "\n\n");
    },
  };
}

export const EXIT_WORDS = new Set(["exit", "quit", "退出"]);

/** Interactive chat until exit / Ctrl+C / end of input. */
export async function chat(ctx: AskContext, session: Session): Promise<Answer[]> {
  const answers: Answer[] = [];
  const tty = process.stdin.isTTY ?? false;
  const rl = createInterface({ input: process.stdin, output: process.stdout, terminal: tty });
  const p = printer(process.stdout, process.stdout, process.stdout.isTTY ?? false);
  rl.on("SIGINT", () => rl.close()); // Ctrl+C: end the loop below, then exit normally
  let closed = false;
  rl.on("close", () => (closed = true)); // piped input ends early; its lines are still read below
  process.stdout.write("问这局的任何决定（输入 exit 结束）。\n");
  rl.setPrompt("\n你：");
  rl.prompt();
  try {
    // The line iterator buffers what arrives while an answer is being written (typed ahead or piped in).
    for await (const line of rl) {
      const question = line.trim();
      if (!tty) process.stdout.write(question + "\n"); // piped input is not echoed
      if (EXIT_WORDS.has(question.toLowerCase())) break;
      if (question) {
        const a = await ask(ctx, session, question, (e) => p.event(e));
        p.done(a, true);
        answers.push(a);
      }
      if (!closed) rl.prompt();
    }
  } finally {
    rl.close();
  }
  return answers;
}

/** Non-interactive: answers to stdout, progress to stderr (no streaming, so stdout is only the answers). */
export async function askAll(ctx: AskContext, session: Session, questions: string[], quiet = false): Promise<Answer[]> {
  const answers: Answer[] = [];
  const p = printer(quiet ? { write: () => {} } : process.stderr, { write: () => {} }, process.stderr.isTTY ?? false);
  for (const question of questions) {
    if (!quiet) process.stderr.write(`\n问：${question}\n`);
    const a = await ask(ctx, session, question, (e) => p.event(e));
    p.done(a, false);
    if (!quiet) process.stdout.write(`问：${question}\n答：${shownAnswer(a)}\n\n`);
    answers.push(a);
  }
  return answers;
}

/** Questions from a file: a JSON array of strings, or one question per line. */
export function parseQuestions(text: string): string[] {
  const trimmed = text.trim();
  if (trimmed.startsWith("[")) return (JSON.parse(trimmed) as unknown[]).map(String).filter((q) => q.trim());
  return trimmed.split(/\r?\n/).map((q) => q.trim()).filter((q) => q && !q.startsWith("#"));
}
