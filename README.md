# MCP Robot — LeKiwi LLM Agent Control System

Control a LeKiwi mobile manipulator robot (or SO-ARM100) using an LLM agent connected via the **Model Context Protocol (MCP)**. Give the robot natural language instructions and watch it execute them in the real world.

---

## What is MCP and How Does It Work?

**Model Context Protocol (MCP)** is an open standard that lets LLMs interact with external tools and systems in a structured way. In this project, it connects an AI agent to a physical robot.

```
┌─────────────────────────────────────────────────────────────────┐
│                        How it works                             │
│                                                                 │
│   You (natural language)                                        │
│       │                                                         │
│       ▼                                                         │
│   ┌────────┐    MCP Protocol    ┌─────────────┐                 │
│   │ Agent  │ ◄────────────────► │ MCP Server  │                 │
│   │ (LLM)  │                    │  (Tools)    │                 │
│   └────────┘                    └──────┬──────┘                 │
│                                        │                        │
│                                        ▼                        │
│                               ┌─────────────────┐              │
│                               │  LeKiwi Host    │              │
│                               │ (Hardware layer)│              │
│                               └─────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

- **Agent** — The AI brain. Receives your instructions, reasons about what to do, and decides which tools to call.
- **MCP Server** — Exposes robot capabilities as callable tools. The agent calls these tools to perform physical actions.
- **LeKiwi Host** — The low-level hardware interface. Controls the actual servos, wheels, and camera.

### Available MCP Tools

| Tool | Description |
|---|---|
| `move_robot` | Move the arm joints to a target pose |
| `move_rover` | Drive the omni-wheel base (forward, backward, rotate) |
| `control_gripper` | Open or close the gripper |
| `get_robot_state` | Read current joint positions and robot state |
| `get_initial_instructions` | Fetch the system prompt / task instructions |

---

## System Architecture (3-Machine Setup)

```
┌─────────────────────────────────────────────────────────┐
│                    Raspberry Pi                         │
│                                                         │
│  Terminal 1: lekiwi_host    ← hardware control layer   │
│  Terminal 2: MCP Server     ← tools exposed via SSE    │
│  Terminal 3: Agent          ← LLM reasoning + calls    │
└─────────────────────────────────────────────────────────┘
         │
         │ WiFi (optional — for local Ollama inference)
         │
┌─────────────────────────────────────────────────────────┐
│              RTX 4090 / GPU Machine (optional)          │
│  ollama serve + qwen2.5:32b or any other model         │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

### Step 1 — Clone and Install LeRobot (Hardware Layer)

LeRobot is the HuggingFace library that controls the LeKiwi hardware.

```bash
cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot
python -m venv .venv
source .venv/bin/activate
pip install -e ".[lekiwi]"
```

> Follow the [official lerobot instructions](https://github.com/huggingface/lerobot) if you run into issues — the Pi may need additional system dependencies.

---

### Step 2 — Clone This Repo and Install Dependencies

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/robot_MCP.git
cd robot_MCP
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> For simplicity this project uses plain `pip` instead of `uv` (often recommended in MCP tutorials) — it works just fine.

If lerobot is not picked up automatically, install it separately using the [official instructions](https://github.com/huggingface/lerobot).

---

## Quick Start

### 1. Connect Your Robot

- Connect **SO-ARM100** via USB
- Update `config.py` with your serial port for SO-ARM (e.g. `/dev/tty.usbmodem58FD0168731`) or `robot_ip` for LeKiwi (e.g. `192.168.1.1`)
- Connect cameras and update `config.py` with the correct indices and names (for LeKiwi, only names matter)

---

### 2. Check Robot Status and Calibration

```bash
python3 check_positions.py
```

This shows the current robot state without sending any commands. Move the robot manually to verify it is properly calibrated.

> After the latest lerobot update, joint states are normalized instead of degrees. Update `MOTOR_NORMALIZED_TO_DEGREE_MAPPING` in `config.py` to match your calibration — you'll need to redo this every time you recalibrate.

---

### 3. Manual Keyboard Control (Test First)

```bash
python3 keyboard_controller.py
```

Control the robot manually with the keyboard. Always test this before using the MCP agent — it confirms your hardware and config are working correctly.

---

### 4. MCP Server in Dev Mode (Debug)

```bash
mcp dev mcp_robot_server.py
```

Opens the MCP Inspector UI so you can test tool calls manually before running the agent. A good final sanity check before going fully autonomous.

---

## Configuration

Create a `.env` file in the project root (`~/robot_MCP/.env`) with your API keys:

```env
# API Keys (at least one required)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# MCP Server Configuration (optional)
MCP_SERVER_IP=127.0.0.1
MCP_PORT=3001
```

You only need to fill in the keys for the providers you actually use.

### Using Ollama (Local Inference — No API Key Needed)

If you have a GPU machine on the same WiFi network, you can run inference locally with Ollama:

**On the GPU machine:**

```bash
ollama serve
ollama pull qwen2.5:32b   # or any model you prefer
```

**On the Pi — point the agent at your GPU machine's IP:**

```bash
python3 agent.py --model ollama/qwen2.5:32b --mcp-server-ip 192.168.1.X
```

Replace `192.168.1.X` with your GPU machine's local IP address. No API key required.

> Check `~/robot_MCP/llm_providers/` to confirm the Ollama provider is present before using `ollama/` models.

---

## Running the System

Open **3 terminals** on the Pi and start them **in this order**:

### Terminal 1 — LeKiwi Host (Hardware Layer)

```bash
cd ~/lerobot
source .venv/bin/activate
python3 -m lerobot.robots.lekiwi.lekiwi_host
```

Starts the low-level hardware interface — servos, wheels, gripper, and camera feed.

---

### Terminal 2 — MCP Server

```bash
cd ~/robot_MCP
source .venv/bin/activate
mcp run mcp_robot_server.py 
```

Starts the MCP server on port `3001` using SSE (Server-Sent Events) transport. Exposes all robot tools to the agent.

---

### Terminal 3 — LLM Agent

**Basic usage:**

```bash
cd ~/robot_MCP
source .venv/bin/activate
python3 agent.py
```

**Advanced usage:**

```bash
# Use Gemini instead of Claude
python3 agent.py --model gemini-2.5-flash

# Override API key
python3 agent.py --api-key your_api_key_here

# Enable image viewer window
python3 agent.py --show-images

# Increase thinking budget for better reasoning
python3 agent.py --thinking-budget 2048

# Custom MCP server location
python3 agent.py --mcp-server-ip 192.168.1.100 --mcp-port 3002
```

---

## Supported Models

### Claude (Anthropic) — Default

- `claude-3-7-sonnet-latest` *(default)*
- All Claude models support thinking, streaming, and multimodal tool results

### Gemini (Google)

- `gemini-2.5-flash`
- `gemini-2.5-pro`
- Use 2.5+ models — they support the thinking feature

### GPT (OpenAI)

- `gpt-4o` and variants
- Most other GPT models don't support thinking or tool calling well — results may vary

### Ollama (Local)

- `ollama/qwen2.5:32b` *(recommended for local inference)*
- Any model available via `ollama list`

---

## Agent Parameters

| Parameter | Default | Description |
|---|---|---|
| `--model` | `claude-3-7-sonnet-latest` | LLM model to use |
| `--api-key` | from `.env` | API key override |
| `--show-images` | off | Display robot camera images in a window |
| `--thinking-budget` | `1024` | Thinking tokens budget (0 to disable) |
| `--thinking-every-n` | `3` | Use thinking every N steps |
| `--mcp-server-ip` | `127.0.0.1` | MCP server IP address |
| `--mcp-port` | `3001` | MCP server port |

---

## Cost Considerations

- **Claude** — counts MCP images in input tokens (higher cost for vision tasks)
- **Gemini** — does not count MCP images in tokens (only text token usage is displayed)
- **Thinking tokens** — add to cost but significantly improve reasoning quality for complex tasks
- **Ollama** — completely free, runs locally on your own hardware

---

## MCP Inspector (Dev / Debug)

To inspect MCP tool calls visually from your laptop, use SSH port forwarding:

```bash
ssh -L 6274:localhost:6274 -L 6277:localhost:6277 pi@<PI_IP>
```

Then open `http://localhost:6274` in your browser.

---

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| `tool_call_id: "unknown"` 400 error | Fixed in OpenAI and Ollama providers |
| VRAM exhaustion during Ollama runs | Disabled automatic image capture on every tool call |
| RealSense camera segfault / timeout on Pi | RealSense init is now guarded with a timeout |

---

## Project Structure

```
~/robot_MCP/
├── mcp_robot_server.py          # MCP server — exposes robot tools
├── agent.py                     # LLM agent — reasoning + tool calls
├── config.py                    # Robot config: serial port, IP, camera indices
├── check_positions.py           # Read robot state without control
├── keyboard_controller.py       # Manual keyboard control for testing
├── requirements.txt             # Python dependencies
├── llm_providers/               # Backend adapters
│   ├── anthropic_provider.py
│   ├── gemini_provider.py
│   ├── openai_provider.py
│   └── ollama_provider.py       # Local inference via Ollama
├── .env                         # API keys (never commit this)
└── .venv/                       # Python virtual environment

~/lerobot/                       # HuggingFace LeRobot (hardware layer)
└── .venv/                       # Separate venv for lerobot
```

---

## Quick Reference

```bash
# Terminal 1 — Hardware host
cd ~/lerobot && source .venv/bin/activate
python3 -m lerobot.robots.lekiwi.lekiwi_host

# Terminal 2 — MCP server
cd ~/robot_MCP && source .venv/bin/activate
mcp run mcp_robot_server.py 

# Terminal 3 — Agent (pick your model)
cd ~/robot_MCP && source .venv/bin/activate
python3 agent.py                                          # Claude (default)
python3 agent.py --model gemini-2.5-flash                # Gemini
python3 agent.py --model ollama/qwen2.5:32b              # Local Ollama
```
