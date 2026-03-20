## Summary

This PR fixes several issues in `src/serena/tools/symbol_tools.py`:

### 1. Fix hover_info key mismatch (core bug fix)
`request_info_for_symbol_batch()` returns `dict[LanguageServerSymbol, str | None]`, but the lookup via `hover_info.get(symbol)` in `_enhance_symbol_dict` failed due to object identity mismatch — the symbol instances returned as dictionary keys are not the same objects passed during lookup.

**Solution:** Convert the hover_info dict to use stable string keys via a new `_hover_key()` helper function that generates keys from `(path:line:column)`, falling back to `(name:kind)` when location data is absent.

### 2. Remove all temporary debug `print()` statements
The `[DEBUG_HOVER]` print statements were writing to **stdout**, which in Serena's MCP server context **pollutes the MCP protocol stream** (stdout is used for MCP communication). Serena has a proper logging infrastructure:
- `MemoryLogHandler` → dashboard/GUI/file logs
- Logs accessible via dashboard tab "Logs", GUI log window, or `~/.serena/logs/`

All debug print statements have been **removed entirely**.

### 3. Fix broken f-string syntax
Line ~88 had a missing closing quote on the f-string in the `ValueError` raise, which would cause a `SyntaxError` at import time.

### 4. Minor formatting
- frozenset literals reformatted to single-line (cosmetic, black-compatible)
- Unicode characters in comments replaced with ASCII equivalents

### Logging note
All logging uses Serena's proper `logging` infrastructure — no `print()` calls remain. Debug messages will be visible in the dashboard, GUI log window, and log files when `log_level` is set to `DEBUG` (10) in `~/.serena/serena_config.yml` or via `--log-level DEBUG` CLI flag.