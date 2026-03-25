import * as esbuild from "esbuild";
import { readFileSync } from "fs";

const watch = process.argv.includes("--watch");

// Inline the CSS as a string so we can inject it into Shadow DOM
const cssText = readFileSync("styles/widget.css", "utf8");

/** @type {import('esbuild').BuildOptions} */
const opts = {
  entryPoints: ["src/widget.ts"],
  bundle: true,
  minify: !watch,
  format: "iife",
  target: "es2020",
  outfile: "dist/widget.js",
  define: {
    __WIDGET_CSS__: JSON.stringify(cssText),
  },
};

if (watch) {
  const ctx = await esbuild.context(opts);
  await ctx.watch();
  console.log("watching...");
} else {
  await esbuild.build(opts);
  const { size } = await import("fs").then((fs) =>
    fs.promises.stat("dist/widget.js")
  );
  console.log(`built dist/widget.js (${(size / 1024).toFixed(1)} KB)`);
}
