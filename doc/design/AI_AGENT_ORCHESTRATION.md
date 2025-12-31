# AI Agent Orchestration Design Plan

**Feature**: Multi-model routing and graduated supervision for CENTAUR
**Status**: Design Document
**Created**: 2025-12-30

---

## Executive Summary

CENTAUR is designed from the ground up to support AI agents of differing capabilities operating at different supervision levels. This document formalizes the architectural vision for:

1. **Graduated Supervision**: A five-level spectrum from fully human-operated to fully autonomous execution, allowing users to calibrate the degree of AI autonomy appropriate to their context, trust level, and task sensitivity.

2. **Multi-Model Routing**: A provider-agnostic system for routing tasks to models of appropriate capability, optimizing the tradeoff between cost, latency, and reasoning depth.

3. **Operator Patterns**: Support for multiple operator types—human researchers, AI assistants (like Claude Code), and autonomous agents—each with appropriate interfaces and safety guarantees.

The core insight is that AI-assisted research exists on a spectrum. A PhD student learning the pipeline needs high supervision; a routine figure regeneration can run autonomously. A simple file parse doesn't need frontier model capabilities; a synthetic peer review demands sophisticated reasoning. CENTAUR's architecture accommodates this entire spectrum without forcing users into a single mode of operation.

This design is **intentionally forward-looking**. While not all features are implemented today, the architecture anticipates a future where AI agents are primary operators of research pipelines, with humans providing direction, oversight, and judgment at key decision points.

---

## Architecture Overview

The orchestration layer sits between operator intent and pipeline execution, managing two orthogonal dimensions: **who decides** (supervision) and **what capability is needed** (model routing).

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OPERATOR LAYER                                  │
│                                                                              │
│    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│    │   Human     │    │     AI      │    │ Autonomous  │                    │
│    │  Operator   │    │  Assistant  │    │    Agent    │                    │
│    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                    │
│           │                  │                  │                            │
│           └──────────────────┼──────────────────┘                            │
│                              │                                               │
│                              ▼                                               │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │                    ORCHESTRATION LAYER                               │  │
│    │                                                                      │  │
│    │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │  │
│    │  │  Supervision   │  │     Model      │  │     Task       │         │  │
│    │  │    Manager     │  │    Router      │  │   Classifier   │         │  │
│    │  │                │  │                │  │                │         │  │
│    │  │ • Level config │  │ • Tier select  │  │ • Complexity   │         │  │
│    │  │ • Approvals    │  │ • Provider map │  │ • Category     │         │  │
│    │  │ • Escalation   │  │ • Fallbacks    │  │ • Heuristics   │         │  │
│    │  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘         │  │
│    │          │                   │                   │                  │  │
│    │          └───────────────────┼───────────────────┘                  │  │
│    │                              │                                      │  │
│    │                              ▼                                      │  │
│    │          ┌───────────────────────────────────────┐                  │  │
│    │          │         Execution Controller          │                  │  │
│    │          │   • Pre-execution checks              │                  │  │
│    │          │   • Audit logging                     │                  │  │
│    │          │   • Post-execution validation         │                  │  │
│    │          └───────────────────┬───────────────────┘                  │  │
│    │                              │                                      │  │
│    └──────────────────────────────┼──────────────────────────────────────┘  │
│                                   │                                         │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PIPELINE LAYER                                    │
│                                                                              │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐            │
│  │  s00   │──►│  s01   │──►│  s02   │──►│  s03   │──►│  ...   │            │
│  │ ingest │   │  link  │   │ panel  │   │ estim  │   │        │            │
│  └────────┘   └────────┘   └────────┘   └────────┘   └────────┘            │
│                                                                              │
│  Each stage can have:                                                        │
│  • Supervision override (e.g., s03 requires human approval)                  │
│  • Tier override (e.g., s07 always uses complex tier)                        │
│  • Custom escalation rules                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Orthogonal Concerns**: Supervision level (who decides) and model tier (what capability) are independent. You can have autonomous execution with a simple model, or human-approved execution with a complex model.

2. **Defaults with Overrides**: Session-level defaults cover most cases; per-stage overrides handle exceptions. This minimizes configuration while preserving flexibility.

3. **Graceful Degradation**: If a preferred model tier is unavailable, the system falls back to the next available tier rather than failing.

4. **Audit Everything**: All supervision decisions, model selections, and escalations are logged for reproducibility and debugging.

---

## Supervision Levels

CENTAUR defines five supervision levels representing the spectrum from fully human-controlled to fully autonomous operation.

### Level Definitions

| Level | Name | Description |
|-------|------|-------------|
| 0 | **Fully Human** | Human executes all operations manually. AI provides documentation and explanations only. |
| 1 | **AI-Assisted** | AI suggests actions and provides recommendations. Human makes all decisions and executes. |
| 2 | **Human-Approved** | AI proposes specific actions. Human reviews and approves before execution proceeds. |
| 3 | **Supervised** | AI executes autonomously. Human monitors progress and can intervene if needed. |
| 4 | **Autonomous** | AI executes independently. Human is notified of completion and handles exceptions only. |

### Detailed Level Descriptions

#### Level 0: Fully Human

The traditional mode of operation. CENTAUR serves as a well-documented CLI tool that humans operate directly.

**Human responsibilities:**
- Read documentation to understand available commands
- Decide which stages to run and in what order
- Execute all pipeline commands manually
- Interpret results and make decisions

**AI role:**
- Provide help text and documentation
- Answer questions about the codebase
- Explain error messages

**Use cases:**
- Learning the pipeline for the first time
- Debugging complex issues that require human judgment at every step
- Situations where AI assistance is unavailable or undesired

#### Level 1: AI-Assisted

AI acts as an advisor, suggesting actions but leaving all decisions and execution to the human.

**Human responsibilities:**
- Evaluate AI suggestions critically
- Decide whether to follow recommendations
- Execute chosen commands manually
- Retain full control over all actions

**AI role:**
- Analyze current state and suggest next steps
- Explain tradeoffs between options
- Provide command syntax and examples
- Answer clarifying questions

**Use cases:**
- Users new to the codebase who want guidance
- Complex decisions where multiple valid approaches exist
- Situations requiring domain expertise the AI may lack

#### Level 2: Human-Approved

AI formulates specific actions and requests approval before executing. This is the default for sensitive operations.

**Human responsibilities:**
- Review proposed actions before they execute
- Approve, modify, or reject AI proposals
- Provide feedback on rejected proposals
- Monitor overall direction

**AI role:**
- Analyze context and formulate specific actions
- Present actions clearly with rationale
- Wait for explicit approval before executing
- Incorporate feedback into revised proposals

**Approval patterns:**
- Per-action approval (most conservative)
- Batch approval for related actions
- Conditional approval ("proceed unless X")

**Use cases:**
- Statistical estimation where methodology matters
- Any operation that modifies data
- Operations with cost implications (API calls)
- First run of a new pipeline configuration

#### Level 3: Supervised

AI executes autonomously while human monitors. Human can intervene but doesn't approve each action.

**Human responsibilities:**
- Set overall goals and constraints
- Monitor progress through logs and status updates
- Intervene if something goes wrong
- Review final outputs

**AI role:**
- Execute pipeline stages autonomously
- Report progress and significant events
- Pause and escalate on uncertainty or errors
- Complete tasks without per-action approval

**Monitoring patterns:**
- Real-time log streaming
- Periodic status summaries
- Alerts on anomalies or errors
- Final completion notification

**Use cases:**
- Routine pipeline runs with established configurations
- Batch processing of multiple specifications
- Overnight or background execution
- Tasks within well-understood parameters

#### Level 4: Autonomous

AI executes independently with minimal human involvement. Humans handle exceptions only.

**Human responsibilities:**
- Define success criteria and constraints upfront
- Handle escalated exceptions
- Review periodic summaries
- Maintain the system

**AI role:**
- Execute full pipelines end-to-end
- Make judgment calls within defined parameters
- Escalate only genuine exceptions
- Self-correct minor issues

**Escalation triggers:**
- Errors that cannot be automatically resolved
- Results outside expected bounds
- Resource usage exceeding thresholds
- Decisions requiring domain judgment

**Use cases:**
- Scheduled pipeline runs (nightly builds)
- CI/CD integration
- Large-scale batch processing
- Mature, stable pipelines

### Session vs. Stage Configuration

Supervision levels can be configured at two scopes:

#### Session-Level Default

Set once for the entire session. Applies to all stages unless overridden.

```python
# Example: config.py
ORCHESTRATION = {
    'default_supervision': 'supervised',  # Level 3
}
```

#### Stage-Level Override

Specific stages can require different supervision levels regardless of session default.

```python
# Example: config.py
ORCHESTRATION = {
    'default_supervision': 'supervised',
    'stage_overrides': {
        's03_estimation': {'supervision': 'human-approved'},
        's07_reviews': {'supervision': 'human-approved'},
        's09_writing': {'supervision': 'human-approved'},
    },
}
```

**Rationale for stage overrides:**
- **s03_estimation**: Statistical results are the core of the paper; methodology choices matter
- **s07_reviews**: Synthetic reviews inform revision strategy; judgment required
- **s09_writing**: Prose represents the researcher's voice; human approval essential

### Escalation Patterns

Even at high autonomy levels, certain conditions should trigger escalation to human oversight.

#### Automatic Escalation Triggers

| Trigger | Description | Escalate To |
|---------|-------------|-------------|
| **Error** | Unhandled exception or failure | Human-Approved |
| **Threshold breach** | QA metric outside bounds | Supervised |
| **Resource limit** | API cost or time exceeds limit | Human-Approved |
| **Ambiguity** | Multiple valid interpretations | AI-Assisted |
| **Data anomaly** | Unexpected data patterns | Human-Approved |
| **First occurrence** | Never-before-seen situation | Human-Approved |

#### Escalation Behavior

When escalation is triggered:

1. **Pause execution** at a safe point
2. **Log context** including state, trigger, and options
3. **Notify human** with clear explanation of the situation
4. **Present options** if multiple valid paths exist
5. **Wait for guidance** before proceeding

#### De-escalation

After human intervention resolves an issue:

1. **Log resolution** including human decision
2. **Resume execution** at appropriate supervision level
3. **Learn from pattern** (future implementation: similar situations may not require escalation)

---

## Multi-Model Routing

CENTAUR routes tasks to models of appropriate capability using a provider-agnostic tier system. This optimizes the tradeoff between cost, latency, and reasoning depth.

### Design Philosophy

**Provider-agnostic by design.** The tier system describes capability requirements, not specific models. This allows:

- Swapping providers without reconfiguring the entire system
- Using the best available model at any given time
- Supporting local models, cloud APIs, and hybrid configurations
- Future-proofing against model capability changes

### Capability Tiers

| Tier | Capability Profile | Typical Characteristics |
|------|-------------------|------------------------|
| **Simple** | Pattern matching, formatting, extraction, validation | Fast response (<1s), low cost, deterministic outputs preferred |
| **Standard** | Code generation, analysis, summarization, multi-step reasoning | Moderate response time, moderate cost, good general capability |
| **Complex** | Deep reasoning, creativity, nuanced judgment, long-form generation | Higher latency acceptable, higher cost acceptable, frontier capability |

### Tier Characteristics

#### Simple Tier

**Capability requirements:**
- Reliable pattern matching and extraction
- Consistent output formatting
- Basic logical operations
- High throughput capacity

**Typical tasks:**
- Parsing structured data (CSV, JSON, logs)
- Regex-based extraction
- QA threshold comparisons
- File format validation
- Simple transformations

**Performance expectations:**
- Response time: <1 second
- Cost: minimal per request
- Accuracy: >99% for well-defined tasks
- Throughput: high volume acceptable

#### Standard Tier

**Capability requirements:**
- Code generation and modification
- Multi-step reasoning
- Context synthesis across documents
- Structured analysis and summarization

**Typical tasks:**
- Writing or modifying pipeline stage code
- Analyzing estimation results
- Generating documentation
- Debugging errors with stack traces
- Creating visualizations from specifications

**Performance expectations:**
- Response time: 5-30 seconds acceptable
- Cost: moderate per request
- Accuracy: >90% with human review
- Complexity: multi-file context handling

#### Complex Tier

**Capability requirements:**
- Sophisticated reasoning about methodology
- Creative problem-solving
- Nuanced judgment calls
- Long-form coherent generation
- Domain expertise application

**Typical tasks:**
- Generating synthetic peer reviews
- Drafting results narratives
- Making methodology recommendations
- Evaluating tradeoffs between approaches
- Writing abstracts and conclusions

**Performance expectations:**
- Response time: 30-120 seconds acceptable
- Cost: higher per request acceptable for high-value tasks
- Quality: publication-ready with editing
- Depth: sustained reasoning over long contexts

### Task Classification

The system classifies tasks to determine appropriate tier routing.

#### Classification Heuristics

```text
Task Classification Decision Tree
─────────────────────────────────

Task arrives
    │
    ▼
┌─────────────────────────────────┐
│ Is output format fixed/simple?  │
│ (e.g., boolean, number, enum)   │
└────────────┬────────────────────┘
             │
        Yes  │  No
             │   │
             ▼   │
      ┌──────────┐   │
      │ SIMPLE   │   │
      │ TIER     │   │
      └──────────┘   │
                     ▼
        ┌────────────────────────────┐
        │ Does task require          │
        │ code generation/editing?   │
        └────────────┬───────────────┘
                     │
                Yes  │  No
                     │   │
                     ▼   │
              ┌──────────┐   │
              │ STANDARD │   │
              │ TIER     │   │
              └──────────┘   │
                             ▼
                ┌────────────────────────┐
                │ Does task require      │
                │ judgment, creativity,  │
                │ or domain expertise?   │
                └────────────┬───────────┘
                             │
                        Yes  │  No
                             │   │
                             ▼   ▼
                      ┌──────────┐  ┌──────────┐
                      │ COMPLEX  │  │ STANDARD │
                      │ TIER     │  │ TIER     │
                      └──────────┘  │ (default)│
                                    └──────────┘
```

#### Classification Signals

| Signal | Simple | Standard | Complex |
|--------|--------|----------|---------|
| Output length | Short (<100 tokens) | Medium (<2000 tokens) | Long (>2000 tokens) |
| Reasoning steps | 1-2 | 3-10 | 10+ |
| Domain knowledge | Generic | Technical | Expert |
| Creativity required | None | Some | Significant |
| Judgment required | None | Limited | Extensive |
| Error tolerance | Low (deterministic) | Medium | Higher (review expected) |

### Task-to-Tier Mapping by Category

| Category | Examples | Default Tier |
|----------|----------|--------------|
| **Parsing** | CSV ingestion, log extraction, JSON parsing | Simple |
| **Validation** | Schema checks, QA thresholds, format verification | Simple |
| **Transformation** | Data cleaning, type conversion, normalization | Simple |
| **Code Generation** | New functions, bug fixes, stage modifications | Standard |
| **Analysis** | Result interpretation, pattern finding, summarization | Standard |
| **Documentation** | README updates, docstrings, inline comments | Standard |
| **Review** | Peer review generation, code critique, methodology review | Complex |
| **Strategy** | Architecture decisions, methodology selection | Complex |
| **Writing** | Abstract drafting, results narrative, discussion sections | Complex |

### Pipeline Stage Defaults

Each pipeline stage has a default tier based on its primary tasks:

| Stage | Primary Tasks | Default Tier | Rationale |
|-------|---------------|--------------|-----------|
| s00_ingest | File parsing, validation | Simple | Deterministic data operations |
| s01_link | Record matching, deduplication | Standard | Fuzzy matching requires reasoning |
| s02_panel | Panel construction, reshaping | Standard | Multi-step transformations |
| s03_estimation | Model execution, result extraction | Standard | Technical but well-defined |
| s04_robustness | Specification variants, placebo tests | Standard | Systematic variations |
| s05_figures | Visualization generation | Standard | Design choices involved |
| s06_validation | Journal requirement checking | Simple | Rule-based validation |
| s07_reviews | Synthetic peer review generation | Complex | Requires domain expertise |
| s08_journal | Submission preparation | Standard | Formatting and assembly |
| s09_writing | Draft generation, editing | Complex | Creative, judgment-intensive |

### Provider Configuration

Tiers map to concrete providers and models via configuration:

```python
# Example: config.py (future implementation)
LLM_TIER_PROVIDERS = {
    'simple': {
        'provider': 'anthropic',
        'model': 'claude-3-haiku',
        'fallback': {'provider': 'openai', 'model': 'gpt-4o-mini'},
    },
    'standard': {
        'provider': 'anthropic',
        'model': 'claude-sonnet',
        'fallback': {'provider': 'openai', 'model': 'gpt-4o'},
    },
    'complex': {
        'provider': 'anthropic',
        'model': 'claude-opus',
        'fallback': {'provider': 'openai', 'model': 'gpt-4-turbo'},
    },
}
```

### Fallback Chains

When a preferred provider is unavailable:

1. **Try primary** provider for tier
2. **Try fallback** provider if configured
3. **Escalate tier** (simple→standard, standard→complex) if no fallback
4. **Fail gracefully** with clear error if no options remain

### Tier Override

Users can override tier selection for specific operations:

```bash
# Force complex tier for estimation (future CLI)
python src/pipeline.py run_estimation --tier complex

# Force simple tier for speed (accepts lower quality)
python src/pipeline.py make_figures --tier simple
```

---

## Operator Patterns

CENTAUR supports multiple operator types, each with appropriate interfaces and interaction patterns.

### Human Operator

The traditional research workflow where a human directly operates the pipeline.

#### Interaction Model

```text
Human ──[types commands]──► CLI ──[executes]──► Pipeline
   ▲                                               │
   └────────────[views output]─────────────────────┘
```

#### Characteristics

- **Control**: Complete control over every action
- **Pace**: Human-determined; can pause indefinitely
- **Context**: Human maintains mental model of project state
- **Learning**: Human learns pipeline through direct use

#### Appropriate When

- Learning the system
- Debugging complex issues
- Making sensitive decisions
- AI assistance unavailable

### AI Assistant (Claude Code, Cursor, etc.)

An AI assistant operates the pipeline on behalf of a human who remains in the conversation loop.

#### Interaction Model

```text
Human ──[natural language]──► AI Assistant ──[commands]──► CLI ──► Pipeline
   ▲                               │                           │
   │                               │                           │
   └──────[conversation]───────────┴───────[results]───────────┘
```

#### Characteristics

- **Control**: Shared between human and AI based on supervision level
- **Pace**: Conversational; AI explains and waits for direction
- **Context**: AI maintains context within conversation
- **Translation**: AI translates intent to specific commands

#### Supervision Integration

| Supervision Level | AI Assistant Behavior |
|-------------------|----------------------|
| Fully Human | AI explains options; human types commands |
| AI-Assisted | AI suggests commands; human decides |
| Human-Approved | AI proposes actions; waits for "yes" |
| Supervised | AI executes; narrates progress |
| Autonomous | AI completes task; reports results |

#### Appropriate When

- Users prefer natural language interaction
- Task requires codebase exploration
- Multiple steps need coordination
- Human wants to stay informed but not execute manually

### Autonomous Agent

A scheduled or triggered agent that executes without real-time human interaction.

#### Interaction Model

```text
Trigger ──► Autonomous Agent ──► Pipeline
(cron,CI,                          │
 webhook)                          ▼
              Human ◄── Notification ◄── Results
                │
                └── Exception Handler (if escalated)
```

#### Characteristics

- **Control**: Agent has full control within defined parameters
- **Pace**: As fast as pipeline allows
- **Context**: Agent loads context from configuration and history
- **Reporting**: Async notifications, not conversation

#### Configuration Requirements

Autonomous agents require explicit configuration:

```python
# Example autonomous agent config
AUTONOMOUS_AGENT = {
    'enabled': True,
    'allowed_stages': ['s00_ingest', 's05_figures'],  # Whitelist
    'forbidden_stages': ['s03_estimation'],  # Blacklist
    'escalation_email': 'researcher@university.edu',
    'max_runtime_minutes': 60,
    'max_api_cost_dollars': 10.00,
    'notification_on_complete': True,
}
```

#### Appropriate When

- Routine, well-defined tasks
- Scheduled updates (nightly data refresh)
- CI/CD integration
- High-volume batch processing

### Multi-Agent Coordination

Multiple AI agents working together, potentially with different roles and capabilities.

#### Interaction Model

```text
┌─────────────────────────────────────────────────────────┐
│                   Coordination Layer                     │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │  Agent A │   │  Agent B │   │  Agent C │            │
│  │ (simple) │   │(standard)│   │ (complex)│            │
│  │          │   │          │   │          │            │
│  │ Parsing  │   │  Code    │   │  Review  │            │
│  │ Tasks    │   │  Gen     │   │  Tasks   │            │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘            │
│       │              │              │                   │
│       └──────────────┼──────────────┘                   │
│                      ▼                                  │
│              ┌──────────────┐                           │
│              │   Pipeline   │                           │
│              └──────────────┘                           │
└─────────────────────────────────────────────────────────┘
```

#### Coordination Patterns

**Sequential handoff**: Agent A completes, hands context to Agent B
```text
Agent A (parse data) ──► Agent B (analyze) ──► Agent C (write)
```

**Parallel execution**: Multiple agents work on independent tasks
```text
Agent A (figures) ─┬─► Merge ──► Pipeline
Agent B (tables)  ─┘
```

**Supervisor pattern**: One agent coordinates others
```text
Supervisor Agent
    │
    ├──► Worker A
    ├──► Worker B
    └──► Worker C
```

#### Conflict Resolution

When agents produce conflicting outputs:

1. **Last-write-wins**: Most recent output takes precedence
2. **Supervisor decides**: Coordinating agent resolves conflicts
3. **Human escalation**: Genuine conflicts escalate to human
4. **Merge where possible**: Non-conflicting changes combined

---

## Safety and Guardrails

Regardless of supervision level or operator type, certain safety measures always apply.

### Always-Human-Approved Operations

Some operations require human approval regardless of supervision level:

| Operation | Rationale |
|-----------|-----------|
| Delete data in `data_raw/` | Raw data is irreplaceable |
| Modify git history | Destructive, affects reproducibility |
| Push to remote repository | Public action, hard to reverse |
| API calls exceeding cost threshold | Financial implications |
| Operations on files outside project | Security boundary |
| Installing dependencies | Supply chain security |

### Irreversible Action Protections

Before executing irreversible actions:

1. **Confirm intent**: "This will permanently delete X. Proceed?"
2. **Create backup**: Snapshot state before destructive operations
3. **Log action**: Record what was done for audit trail
4. **Delay execution**: Brief pause to allow cancellation

### Data Safety Boundaries

```text
PROTECTED (never modify without explicit approval):
├── data_raw/           # Original source data
├── .git/               # Version control history
├── .env                # Secrets and credentials
└── credentials*.json   # API keys

CAUTIOUS (warn before modifying):
├── data_work/          # Processed data (can regenerate)
├── src/                # Pipeline code
└── manuscript_*/       # Manuscript content

SAFE (modify freely):
├── data_work/.cache/   # Computation cache
├── _output/            # Rendered outputs
└── logs/               # Log files
```

### Cost Controls

For operations involving paid API calls:

```python
# Example cost control config
COST_CONTROLS = {
    'warn_threshold_dollars': 5.00,
    'approve_threshold_dollars': 20.00,
    'hard_limit_dollars': 100.00,
    'track_by_stage': True,
    'track_by_session': True,
}
```

**Behavior:**
- **Below warn**: Execute silently
- **Above warn**: Log warning, continue
- **Above approve**: Require human approval
- **Above hard limit**: Refuse execution

### Audit Logging

All orchestration decisions are logged:

```text
data_work/orchestration/audit.jsonl

{"timestamp": "2025-12-30T10:15:00Z", "event": "supervision_decision",
 "stage": "s03_estimation", "level": "human-approved", "operator": "claude-code"}
{"timestamp": "2025-12-30T10:15:05Z", "event": "model_selection",
 "task": "run_estimation", "tier": "standard", "provider": "anthropic"}
{"timestamp": "2025-12-30T10:15:10Z", "event": "human_approval",
 "action": "execute_estimation", "approved": true, "latency_seconds": 5}
```

---

## Configuration Schema

### Full Configuration Example

```python
# src/config.py - Orchestration settings (future implementation)

# ============================================================================
# ORCHESTRATION CONFIGURATION
# ============================================================================

ORCHESTRATION = {
    # Session-level defaults
    'default_supervision': 'supervised',  # Level 3
    'default_tier': 'standard',

    # Per-stage overrides
    'stage_overrides': {
        's03_estimation': {
            'supervision': 'human-approved',
            'tier': 'standard',
        },
        's07_reviews': {
            'supervision': 'human-approved',
            'tier': 'complex',
        },
        's09_writing': {
            'supervision': 'human-approved',
            'tier': 'complex',
        },
    },

    # Escalation settings
    'escalation': {
        'on_error': 'human-approved',
        'on_threshold_breach': 'supervised',
        'on_ambiguity': 'ai-assisted',
        'timeout_seconds': 300,
    },
}

# Model tier to provider mapping
LLM_TIER_PROVIDERS = {
    'simple': {
        'provider': 'local',  # Or 'anthropic' with haiku
        'model': 'local-small',
        'fallback': {'provider': 'anthropic', 'model': 'claude-3-haiku'},
    },
    'standard': {
        'provider': 'anthropic',
        'model': 'claude-sonnet',
        'fallback': {'provider': 'openai', 'model': 'gpt-4o'},
    },
    'complex': {
        'provider': 'anthropic',
        'model': 'claude-opus',
        'fallback': {'provider': 'openai', 'model': 'o1'},
    },
}

# Cost controls
COST_CONTROLS = {
    'enabled': True,
    'warn_threshold_dollars': 5.00,
    'approve_threshold_dollars': 20.00,
    'hard_limit_dollars': 100.00,
    'reset_period': 'session',  # or 'daily', 'weekly'
}

# Autonomous agent settings (for scheduled/triggered runs)
AUTONOMOUS_AGENT = {
    'enabled': False,
    'allowed_stages': ['s00_ingest', 's05_figures', 's06_validation'],
    'max_runtime_minutes': 60,
    'notification_email': None,
}
```

### CLI Integration

```bash
# Override supervision level for a single command
python src/pipeline.py run_estimation --supervision human-approved
python src/pipeline.py run_estimation --autonomous

# Override model tier
python src/pipeline.py draft_results --tier complex
python src/pipeline.py make_figures --tier simple

# Combine overrides
python src/pipeline.py review_new --supervision supervised --tier complex

# Query current configuration
python src/pipeline.py config show orchestration
python src/pipeline.py config show tiers
```

---

## Integration with Existing Architecture

### LLM Provider Abstraction

The existing `src/llm/` module provides the foundation for multi-model routing.

**Current structure:**
```text
src/llm/
├── __init__.py      # Factory function get_provider()
├── base.py          # LLMProvider Protocol
├── anthropic.py     # Claude implementation
└── openai.py        # OpenAI implementation
```

**Extension for tier routing:**
```python
# src/llm/__init__.py extension

def get_provider_for_tier(tier: str) -> LLMProvider:
    """Get appropriate provider for capability tier."""
    tier_config = LLM_TIER_PROVIDERS.get(tier, LLM_TIER_PROVIDERS['standard'])

    try:
        return get_provider(tier_config['provider'], tier_config['model'])
    except ProviderUnavailable:
        if 'fallback' in tier_config:
            return get_provider_for_tier_config(tier_config['fallback'])
        raise
```

### QA Integration

Orchestration events integrate with the existing QA reporting system:

```text
data_work/quality/
├── s00_ingest_quality_20251230_101500.csv     # Stage QA
├── orchestration_audit_20251230.jsonl          # Orchestration log
└── model_usage_20251230.csv                    # Token/cost tracking
```

**QA report additions:**
- Model tier used for each operation
- Supervision level active during execution
- Escalation events and resolutions
- Cost accumulation per stage

### Git Integration

Orchestration metadata in commits:

```text
commit abc123
Author: researcher <researcher@university.edu>
Date:   Mon Dec 30 10:15:00 2025

    Update estimation with new specification

    Orchestration:
    - Supervision: supervised
    - Model tier: standard
    - Operator: claude-code
    - Session: sess_abc123
```

---

## Implementation Roadmap

This design document is Phase 1. Subsequent phases will implement the architecture.

| Phase | Scope | Status | Dependencies |
|-------|-------|--------|--------------|
| **1** | Design documentation | **This document** | None |
| **2** | Configuration schema | Future | Phase 1 |
| **3** | Supervision level CLI flags | Future | Phase 2 |
| **4** | Multi-tier provider routing | Future | Phase 2 |
| **5** | Task complexity classifier | Future | Phase 4 |
| **6** | Audit logging | Future | Phase 3 |
| **7** | Cost controls | Future | Phase 6 |
| **8** | Autonomous agent mode | Future | Phase 3, 6, 7 |
| **9** | Multi-agent coordination | Future | Phase 8 |

### Phase 2: Configuration Schema

Add orchestration configuration to `src/config.py`:
- Supervision level settings
- Tier provider mappings
- Stage overrides
- Cost control thresholds

### Phase 3: Supervision CLI Flags

Add `--supervision` flag to pipeline commands:
- Parse and validate supervision level
- Apply to command execution
- Log supervision decisions

### Phase 4: Multi-Tier Provider Routing

Extend `src/llm/` module:
- Tier-based provider selection
- Fallback chain implementation
- Provider availability checking

### Phase 5: Task Complexity Classifier

Implement automatic tier selection:
- Heuristic classification rules
- Override mechanisms
- Logging of classification decisions

---

## Appendix: Design Decisions

### Why Provider-Agnostic Tiers?

**Decision**: Define tiers by capability requirements, not specific models.

**Rationale**:
1. **Portability**: Users can swap providers without reconfiguring
2. **Future-proofing**: Model capabilities change; tiers remain stable
3. **Cost flexibility**: Use cheaper providers when appropriate
4. **Availability**: Fallback to alternatives when primary unavailable
5. **Local models**: Support on-device models for simple tier

**Trade-off accepted**: Slightly more configuration complexity.

### Why Both Session and Stage-Level Configuration?

**Decision**: Session defaults with per-stage overrides.

**Rationale**:
1. **Simplicity**: Most users set one default and forget
2. **Flexibility**: Sensitive stages can enforce stricter supervision
3. **Safety**: Critical stages (estimation, writing) always require approval
4. **Efficiency**: Non-critical stages can run faster with less oversight

**Trade-off accepted**: Configuration has two levels to understand.

### Why Five Supervision Levels?

**Decision**: Five levels from fully human (0) to autonomous (4).

**Rationale**:
1. **Granularity**: Each level represents a distinct interaction model
2. **Clarity**: Clear progression from more to less human involvement
3. **Practical**: Maps to real usage patterns observed in AI-assisted coding
4. **Extensible**: Sub-levels could be added without restructuring

**Alternatives considered**:
- Three levels (manual/assisted/autonomous): Too coarse
- Ten levels: Unnecessary granularity
- Named levels only: Harder to compare and configure

### Why Default to Supervised (Level 3)?

**Decision**: Default supervision is "supervised" (Level 3), not "human-approved" (Level 2).

**Rationale**:
1. **Efficiency**: Most pipeline operations are routine and safe
2. **Trust**: Users who adopt AI assistants expect some autonomy
3. **Overrides**: Sensitive stages explicitly require approval anyway
4. **Progressive**: Users can always increase supervision if uncomfortable

**Trade-off accepted**: Slightly higher risk than defaulting to level 2.

---

## References

- [CENTAUR Pipeline Documentation](../PIPELINE.md)
- [Architecture Overview](../ARCHITECTURE.md)
- [LLM Provider Implementation](../../src/llm/)
- [QA Reporting System](../PIPELINE.md#qa-reports)
- [Synthetic Review Process](../SYNTHETIC_REVIEW_PROCESS.md)
