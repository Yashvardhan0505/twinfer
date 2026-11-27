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

// What the chat shows for a generated token. Special tokens (<|im_start|> and the like) are
// control tokens, not text, so they are hidden, as the C++ CLI does.
export function shownBytes(tok, id) {
  return tok.isSpecial(id) ? new Uint8Array(0) : tok.tokenBytes(id);
}
