---
name: brainstorming
description: "Facilitate structured design work before implementation. Use when the user asks to brainstorm, shape, or evaluate features, architecture, workflows, product behavior, requirements, or trade-offs, especially when the request is ambiguous, exploratory, or high impact and should be clarified before writing code."
---

# Brainstorming

Guide the user from vague intent to a confirmed design. Stay in design mode until the design is explicitly accepted.

## Operating Rules

- Do not edit code, propose implementation steps, or change behavior while this skill is active.
- Inspect the current project state before proposing solutions. Read only the files needed to understand what already exists.
- Separate facts, assumptions, risks, and open questions.
- Prefer clarity over speed. Apply YAGNI and avoid speculative scope.

## Workflow

### 1. Inspect Current Context First

- Review relevant files, docs, plans, and prior decisions.
- State what already exists, what appears proposed, and what constraints are visible.
- Do not present design options yet.

### 2. Clarify the Problem One Gap at a Time

- Ask one question per turn.
- Prefer short options when they help the user answer quickly. Use open-ended questions only when necessary.
- Cover goal, user value, target users, success criteria, explicit non-goals, and hard constraints.

### 3. Clarify Non-Functional Requirements

Explicitly confirm or propose assumptions for:

- performance and latency
- scale: users, traffic, and data volume
- security and privacy
- reliability and failure tolerance
- ownership, maintenance, and operational complexity

If the user is unsure, propose defaults and label them as assumptions.

### 4. Lock Understanding Before Design

Before presenting any design, provide:

- an understanding summary in 5-7 bullets
- assumptions
- open questions
- notable risks or constraints

Then ask for explicit confirmation. Do not proceed to design until the user confirms or corrects the summary.

### 5. Explore Options

- Present 2-3 viable approaches.
- Lead with the recommended option.
- Compare complexity, extensibility, delivery risk, and maintenance cost.
- Reject unnecessary complexity.

### 6. Present the Design Incrementally

Break the design into small sections and pause for confirmation after each major part. Cover only the sections that matter:

- architecture and boundaries
- components and responsibilities
- data flow and state changes
- APIs, contracts, or configuration
- error handling and edge cases
- test and rollout strategy

### 7. Maintain a Decision Log

Track each confirmed decision with:

- decision
- alternatives considered
- rationale

When the design stabilizes, write or update a durable Markdown design note if the project expects one.

## Exit Criteria

Exit brainstorming mode only after all of the following are true:

- the understanding summary is confirmed
- a design approach is accepted
- assumptions are documented
- major risks are acknowledged
- the decision log is complete enough to guide implementation

If the user tries to jump into implementation before these are true, explain what is still unresolved and continue the design workflow.
