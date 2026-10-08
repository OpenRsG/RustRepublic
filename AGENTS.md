# Project rules

- Do not use the AISandbox MCP (`mcp__aisandbox_*` / `xd://mcp__aisandbox_*`) for this project, including builds, tests, debugging, GUI verification, or delegated work. The owner explicitly rejected it because it is too slow here.
- Run builds, tests, and debugging locally. For GUI verification, use a headless local display; do not steal focus or inject input into the owner's desktop session.
- This is a Bevy bike rewrite, not a verified reproduction of Riders Republic's proprietary algorithms or animation tracks. Keep the retail asset tools separate and original game files read-only.
- Keep proprietary extracts, evidence, local tools and temporary verification artifacts under ignored `.local/` or `.tools/`.
