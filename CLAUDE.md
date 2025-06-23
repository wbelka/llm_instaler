## Prompts

- Add prompt to memory

## Tools

- Use flake8 as profiler

## Code

- Save information about code

## LLM Installer Commands

Main commands:
- `./llm-installer check <model_id>` - Check model compatibility without downloading
- `./llm-installer install <model_id> [--device auto/cuda/cpu/mps] [--dtype auto/float16/float32/int8/int4] [--quantization none/8bit/4bit] [-f/--force]` - Install model
- `./llm-installer list` - List installed models
- `./llm-installer update <model_dir>` - Update scripts and libraries in installed model
- `./llm-installer fix <model_dir> [-r/--reinstall] [--fix-torch] [--fix-cuda]` - Fix dependencies
- `./llm-installer doctor` - Run system diagnostics
- `./llm-installer config` - Show current configuration

Key options:
- `--debug` - Enable debug logging
- `--version` - Show version

For updating installed models with new fixes:
`./llm-installer update /home/wblk/LLM/models/deepseek-ai_Janus-Pro-7B`