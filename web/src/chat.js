// Chat-turn bookkeeping for main.js (same logic as common/chat.h). No DOM code in here, so
// tests/js_unit.mjs can check it under Node.
import { chatml } from "./tokenizer.js";

// Token ids to feed for the next user message. `chat` is { pos, firstTurn, carry }, where carry is
// the last token of the previous reply if the token limit cut that reply off: it was shown but
// never went through the model. Starts over (reset: true) when the turn wouldn't fit in maxSeq.
export function turnIds(tok, chat, text, { maxNew, maxSeq, system }) {
  const first = () => tok.encode(chatml([{ role: "user", content: text }], system));
  let ids = chat.firstTurn
    ? first()
    : tok.encode(`<|im_end|>\n<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`);
  if (chat.carry >= 0) ids = [chat.carry, ...ids];
  if (chat.pos + ids.length + maxNew >= maxSeq) return { ids: first(), reset: true };
  return { ids, reset: false };
}

// The system prompt a ChatML chat template inserts when the conversation doesn't start with one:
// the literal text between "<|im_start|>system\n" and "<|im_end|>" in the Jinja source. Same rule
// as template_system_prompt() in common/chat.h. "" if there is none.
export function templateSystemPrompt(tmpl) {
  if (typeof tmpl !== "string") return "";
  const open = "<|im_start|>system", close = "<|im_end|>";
  for (let at = tmpl.indexOf(open); at >= 0; at = tmpl.indexOf(open, at + 1)) {
    let b = at + open.length;
    if (tmpl.startsWith("\n", b)) b += 1; // a newline inside the Jinja string literal
    else if (tmpl.startsWith("\\n", b)) b += 2; // or its \n escape (how Qwen2.5 writes it)
    else continue;
    const e = tmpl.indexOf(close, b);
    if (e < 0) break;
    const s = tmpl.slice(b, e);
    if (!s || /['"{}]/.test(s)) continue; // template code, not text
    return s.replaceAll("\\n", "\n");
  }
  return "";
}

// Default system prompt from chat_template.jinja or tokenizer_config.json's chat_template (a
// string, or a list of named templates).
export function defaultSystemPrompt(tokenizerConfig, chatTemplateJinja) {
  if (typeof chatTemplateJinja === "string") return templateSystemPrompt(chatTemplateJinja);
  let t = tokenizerConfig ? tokenizerConfig.chat_template : null;
  if (Array.isArray(t)) t = (t.find((x) => x.name === "default") || {}).template;
  return templateSystemPrompt(t);
}

// What the chat shows for a generated token. Special tokens (<|im_start|> and the like) are
// control tokens, not text, so they are hidden, as the C++ CLI does.
export function shownBytes(tok, id) {
  return tok.isSpecial(id) ? new Uint8Array(0) : tok.tokenBytes(id);
}
