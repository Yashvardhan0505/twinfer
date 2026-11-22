// Run the browser tokenizer (web/src/tokenizer.js) under Node.
//   node tests/js_tokenize.mjs path/to/tokenizer.json strings.json   -> JSON list of id lists
import { readFileSync } from "node:fs";

import { Tokenizer } from "../web/src/tokenizer.js";

const [tokPath, stringsPath] = process.argv.slice(2);
const tok = new Tokenizer(JSON.parse(readFileSync(tokPath, "utf8")));
const strings = JSON.parse(readFileSync(stringsPath, "utf8"));
process.stdout.write(JSON.stringify(strings.map((s) => tok.encode(s))));
