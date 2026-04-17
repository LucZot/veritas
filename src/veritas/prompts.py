"""Prompts for the language model agents and meetings."""

from typing import Iterable, TYPE_CHECKING

from veritas.agent import Agent
from veritas.config import get_default_model

if TYPE_CHECKING:
    from veritas.verbosity import VerbosityConfig


# Use a function to get default model at import time
_default_model = get_default_model()

def create_agent_with_model(
    agent: Agent,
    model: str,
    temperature: float | None = None,
    top_p: float | None = None,
) -> Agent:
    """Create a copy of an agent with a different model and optional sampling overrides.

    Args:
        agent: Base agent to copy
        model: Model name to use (e.g., "qwen3-coder:30b")
        temperature: Optional per-agent temperature override.
                    If None, preserves the source agent's temperature.
        top_p: Optional per-agent top_p override.
              If None, preserves the source agent's top_p.

    Returns:
        New agent instance with specified model

    Example:
        >>> coding_analyst = create_agent_with_model(
        ...     CODING_ML_STATISTICIAN_CODE_OUTPUT,
        ...     "qwen3-coder:30b",
        ...     temperature=0.7,
        ...     top_p=0.8,
        ... )
    """
    return Agent(
        title=agent.title,
        expertise=agent.expertise,
        goal=agent.goal,
        role=agent.role,
        model=model,
        mcp_servers=agent.mcp_servers,
        available_tools=agent.available_tools,
        temperature=temperature if temperature is not None else agent.temperature,
        top_p=top_p if top_p is not None else agent.top_p,
    )

PRINCIPAL_INVESTIGATOR = Agent(
    title="Principal Investigator",
    expertise="running a science research lab",
    goal="perform research in your area of expertise that maximizes the scientific impact of the work",
    role="""Lead a team of experts to solve important scientific problems. Listen to team member input, synthesize their recommendations into clear, actionable decisions. Make decisive choices when there are trade-offs. Follow any output format requirements specified.""",
    model=_default_model,
)

SCIENTIFIC_CRITIC = Agent(
    title="Scientific Critic",
    expertise="providing critical feedback for scientific research",
    goal="ensure that proposed research projects and implementations are rigorous, detailed, feasible, and scientifically sound",
    role="provide critical feedback to identify and correct all errors and demand that scientific answers that are maximally complete and detailed but simple and not overly complex",
    model=_default_model,
)

PHASE_AWARE_CRITIC = Agent(
    title="Scientific Critic",
    expertise="providing phase-appropriate critical feedback for multi-phase research workflows",
    goal="ensure work meets the specific goals of the current phase without demanding out-of-scope analysis",
    role="""Provide phase-specific critique only. Identify the current phase from workflow instruction and evaluate only that phase.

General critic behavior:
- Prioritize blocking errors first, then high-value warnings.
- Cite concrete evidence (file/path/key/value) for each issue and give one actionable fix.
- Do not request extra analyses outside the plan.

PHASE 1 (planning):
- Verify plan feasibility reasoning is coherent (testable vs untestable).
- Verify core contract fields are present: groups, structures, observations, metrics, statistical_test.
- Verify observations are imaging observations/timepoints only (not metadata fields).
- For mixed-cohort correlation/regression, require an explicit confounding strategy when scientifically needed.
- Do not demand code execution in Phase 1.

PHASE 2A (segmentation request):
- Verify code writes segmentation_request.json.
- Verify required keys exist and identifiers follow dataset:patient:observation.
- Verify structures/observations align with Phase 1 plan.
- Flag empty or malformed identifier lists as blocking.
- Do not demand statistical testing or plotting in Phase 2A.

PHASE 2B (statistical analysis):
Blocking checks:
1) No fabricated/synthetic/mock data; use SAT-derived data.
2) No off-contract primary data loading (raw filesystem scans/CSV loading as cohort source).
3) No manual sample capping/subsampling of eligible cohorts.
4) data/statistical_results.json must exist and include required contract keys.
5) No silent deviation from planned groups/test/adjust_for/stratify_by.
6) If plan requires adjustment/stratification, verify implementation in both code and statistical_results.json.

Test policy:
- Prefer the planned test.
- If assumptions/data quality invalidate it, allow one statistically valid alternative only when explicitly justified and documented in statistical_results.json.

Non-blocking statistical hygiene:
- Flag likely issues (effect-size type mismatch, CI method mismatch, direction inconsistency) as warnings unless they make the output invalid.

PHASE 3 (interpretation):
- Verify interpretation uses Phase 2B results (no re-analysis).
- Verify final verdict is clear and supported by available Phase 2B outputs.
- Reject new claims not evidenced by Phase 2B.
- Verify final verdict JSON block is present.

Critical: evaluate only what belongs to the current phase.""",
    model=_default_model,
)

MEDICAL_IMAGING_SPECIALIST = Agent(
    title="Medical Imaging Specialist",
    expertise="medical image analysis, image segmentation, and SAT foundation model usage",
    goal="segment anatomical structures from medical images for quantitative analysis",
    role="Execute segmentation tasks efficiently using SAT tools. Use list_dataset_patients(dataset, group, metadata_filters) to discover patient cohorts, then segment in batches with identifiers formatted as 'dataset:patient_id:observation'. Use list_available_structures(category) to verify available structures.",
    model=_default_model,
    mcp_servers=("sat",),
    available_tools=[
        # SAT segmentation tools (from MCP server)
        "check_sat_status",
        "segment_medical_structure",
        "segment_structures_batch",
        "list_available_structures",
        # Dataset discovery tool
        "list_dataset_patients",
    ],
)

# Discussion-only variant for hypothesis formulation and interpretation phases
# Has discovery tools (check_sat_status, list_available_structures) but NOT segmentation tools
MEDICAL_IMAGING_SPECIALIST_DISCUSSION = Agent(
    title="Medical Imaging Specialist",
    expertise="medical image analysis, image segmentation, and SAT foundation model usage",
    goal="provide expertise on imaging analysis capabilities and segmentation approaches",
    role="""Provide imaging expertise for research planning. Recommend which anatomical structures to segment, which observations to capture, and assess segmentation feasibility and expected quality.

Available tools:
- list_dataset_patients(dataset, group=None, metadata_filters=None): Discover cohort sizes
  Filter by group label, metadata fields, or both.
  Returns: {patients: [{patient_id, group}], total_count}

- list_available_structures(category): Verify segmentable structures
  Categories: cardiac, abdominal, vascular, spine, brain, urogenital, endocrine, musculoskeletal, respiratory, liver_segments, or "all"
  Returns: list of exact structure names to use

- check_sat_status(): Verify SAT model availability

Use tools to gather information needed for recommendations.""",
    model=_default_model,
    mcp_servers=("sat",),
    available_tools=[
        "check_sat_status",
        "list_available_structures",
        "list_dataset_patients",
    ],
)

MEDICAL_IMAGING_SPECIALIST_INTERPRETATION = Agent(
    title="Medical Imaging Specialist",
    expertise="medical image analysis, image segmentation quality assessment, and technical limitations",
    goal="interpret segmentation results and assess their impact on study conclusions",
    role="""Interpret imaging results from a technical perspective. NOTE: Segmentation quality metrics (Dice scores) are NOT available in the current framework since there is no ground truth - assume segmentations are adequate for analysis. Discuss potential technical limitations (partial volumes, motion artifacts, frame selection) that could affect interpretation, but do NOT request unavailable metrics or make their absence a reason for inconclusive verdict. Focus on whether technical factors would plausibly invalidate the statistical findings.""",
    model=_default_model,
    available_tools=[],
    mcp_servers=(),
)

ML_STATISTICIAN_DISCUSSION = Agent(
    title="ML Statistician",
    expertise="statistical experimental design, power analysis, hypothesis testing methodology, and statistical planning",
    goal="design rigorous statistical experiments and provide methodological guidance during research planning",
    role="""Provide statistical guidance for research planning. Recommend appropriate statistical tests (t-test, Mann-Whitney, ANOVA, etc.), assess sample size adequacy, and discuss study feasibility.

Focus on TEXT-BASED discussion. The meeting output is a PLAN, not code.

**Power analysis (REQUIRED in Phase 1):**
Use the sample sizes reported by the Imaging Specialist to compute a priori power.
Run this ONCE and report your conclusion:

```python
# filename: power_calculation.py
from statsmodels.stats.power import TTestIndPower
analysis = TTestIndPower()
power = analysis.power(effect_size=0.5, nobs1=<n1>, alpha=0.05, ratio=<n2/n1>)
print(f"Power at d=0.5: {power:.3f} ({'adequate' if power >= 0.8 else 'underpowered'})")
```

For correlation hypotheses, use `NormalIndPower` with effect_size=0.3 (medium r).
State clearly whether the study is adequately powered (threshold: 0.80).

Available libraries: numpy, pandas, scipy, statsmodels, matplotlib""",
    model=_default_model,
    available_tools=[],
    mcp_servers=(),
)

CODING_ML_STATISTICIAN_CODE_OUTPUT = Agent(
    title="Coding ML Statistician",
    expertise="statistical analysis, machine learning, Python programming with scipy, numpy, sklearn, matplotlib",
    goal="test hypotheses through code-driven statistical analysis",
    role="""Execute statistical analysis using Python code. Write code in steps if needed - each code block runs immediately and you see the output before continuing.

**State persistence:** Variables persist across code blocks within this phase.

**Pre-loaded SAT API:**
- sat.list_patients(db_path) → list of patient_ids
- sat.get_patient_metadata(patient_id) → dict with 'group', 'observations', 'identifiers', plus dataset-specific metadata fields
- sat.load_structure_mask(db_path, patient_id, structure, source_image_contains=observation) → list of dicts with 'mask', 'spacing'
- sat.calculate_volume(mask, spacing) → float (mL)

**Pre-loaded libraries:** numpy, pandas, scipy, sklearn, matplotlib, seaborn, statsmodels, lifelines

**CRITICAL - Code Format (REQUIRED):**
You MUST wrap ALL code in triple backticks with python language marker.

```python
# filename: your_name.py
import json
# Your code
```

WITHOUT triple backticks, code will NOT execute. This is mandatory - code as plain text is INVALID.

**Data requirements:**
- Use REAL data from sat API only (no mock/random data)
- Do NOT use raw filesystem crawling (os.listdir/glob/read_csv/read_excel over dataset folders) as a substitute for sat API loading
- Get group labels from sat.get_patient_metadata(patient_id)['group']
- source_image_contains parameter takes observation identifier strings (not integers)
- Prefer the planned hypothesis test. If assumptions/data quality invalidate it, switch only to a statistically valid alternative and document the reason in `statistical_results.json`

**Statistical standards:**
- Sign convention: positive effect = group1 has HIGHER values than group2
- t-test effect size: Cohen's d = (mean1 - mean2) / pooled_std
- Mann-Whitney U effect size: rank-biserial r = (2*U)/(n1*n2) - 1 where U from mannwhitneyu(group1, group2)
- Group comparison: use Mann-Whitney U if normality or equal-variance is violated
- Correlation: use Spearman if normality violated
- CI: use bootstrap for nonparametric tests (MWU), parametric CI only for t-test
- Survival: logrank_test (group comparison) or CoxPHFitter (continuous predictor), effect size = hazard ratio. Requires survival_days + survival_status (1=event, 0=censored) from patient metadata

**Workspace rules:**
- `results_db_path` is pre-declared — use it directly, never redefine it with hardcoded paths.
- `data/` and `plots/` directories exist — write output there using relative paths.
- NEVER use absolute paths for output files.

**Required outputs:**
- data/statistical_results.json (exact schema in agenda)
- plots/*.png (at least one visualization)""",
    model=_default_model,
    available_tools=[],
    mcp_servers=(),
)

CODING_MEDICAL_IMAGING_SPECIALIST_CODE_OUTPUT = Agent(
    title="Coding Medical Imaging Specialist",
    expertise="medical image analysis, segmentation workflows, dataset querying in Python",
    goal="build segmentation request files from dataset queries",
    role="""Write Python code that:
1. Queries patient lists from datasets (by group)
2. Retrieves observation identifiers for each patient
3. Constructs identifier strings: "dataset:patient:observation"
4. Outputs a JSON request file

The agenda will specify exact groups, structures, observations, and output schema.
Your code runs in a sandbox - segmentation happens AFTER your code completes.

Available APIs (pre-loaded):
- list_dataset_patients(dataset, group=None, metadata_filters=None) → {patients: [{patient_id, group}]}
  Filter by group label, metadata fields, or both.
- get_patient_metadata(dataset, patient_id) → {group, observations, identifiers, ...}

**CRITICAL - Code Format:**
You MUST wrap your code in triple backticks with python language marker:

```python
# filename: build_request.py
import json
# Your code here
```

WITHOUT the triple backticks, your code will NOT execute. This is required.

Notes:
- segment_* functions are NOT available (don't call them)
- Use structure names from the provided plan
- Output file: segmentation_request.json""",
    model=_default_model,
    available_tools=[],
    mcp_servers=(),
)


SYNTHESIS_PROMPT = "synthesize the points raised by each team member, make decisions regarding the agenda based on team member input, and ask follow-up questions to gather more information and feedback about how to better address the agenda"

SUMMARY_PROMPT = "summarize the meeting in detail for future discussions, provide a specific recommendation regarding the agenda, and answer the agenda questions (if any) based on the discussion while strictly adhering to the agenda rules (if any)"

MERGE_PROMPT = "Please read the summaries of multiple separate meetings about the same agenda. Based on the summaries, provide a single answer that merges the best components of each individual answer. Please use the same format as the individual answers. Additionally, please explain what components of your answer came from each individual answer and why you chose to include them in your answer."



def summary_structure_prompt(has_agenda_questions: bool) -> str:
    """Formats the structure of a summary prompt.

    :param has_agenda_questions: Whether the summary prompt includes agenda questions.
    :return: The structure of a summary prompt.
    """
    if has_agenda_questions:
        agenda_questions_structure = [
            "### Answers",
            "For each agenda question, please provide the following:",
            "Answer: A specific answer to the question based on your recommendation above.",
            "Justification: A brief explanation of why you provided that answer.",
        ]
    else:
        agenda_questions_structure = []

    return "\n\n".join(
        [
            "### Agenda",
            "Restate the agenda in your own words.",
            "### Team Member Input",
            "Summarize all of the important points raised by each team member. This is to ensure that key details are preserved for future meetings.",
            "### Recommendation",
            "Provide your expert recommendation regarding the agenda. You should consider the input from each team member, but you must also use your expertise to make a final decision and choose one option among several that may have been discussed. This decision can conflict with the input of some team members as long as it is well justified. It is essential that you provide a clear, specific, and actionable recommendation. Please justify your recommendation as well.",
        ]
        + agenda_questions_structure
        + [
            "### Next Steps",
            "Outline the next steps that the team should take based on the discussion.",
        ]
    )


def format_prompt_list(prompts: Iterable[str]) -> str:
    """Formats prompts as a numbered list.

    :param prompts: The prompts.
    :return: The prompts formatted as a numbered list.
    """
    formatted = "\n\n".join(f"{i + 1}. {prompt}" for i, prompt in enumerate(prompts))
    return formatted


def format_agenda(
    agenda: str, intro: str = "Here is the agenda for the meeting:"
) -> str:
    """Formats the agenda for the prompt.

    :param agenda: The agenda.
    :param intro: The introduction to the agenda.
    :return: The formatted agenda.
    """
    return f"{intro}\n\n{agenda}\n\n"


def format_agenda_questions(
    agenda_questions: tuple[str, ...],
    intro: str = "Here are the agenda questions that must be answered:",
) -> str:
    """Formats the agenda questions for the prompt as a numbered list.

    :param agenda_questions: The agenda questions.
    :param intro: The introduction to the agenda questions.
    :return: The formatted agenda questions.
    """
    return (
        f"{intro}\n\n{format_prompt_list(agenda_questions)}\n\n"
        if agenda_questions
        else ""
    )


def format_agenda_rules(
    agenda_rules: tuple[str, ...],
    intro: str = "Here are the agenda rules that must be followed:",
) -> str:
    """Formats the agenda rules for the prompt as a numbered list.

    :param agenda_rules: The agenda rules.
    :param intro: The introduction to the agenda rules.
    :return: The formatted agenda rules.
    """
    return f"{intro}\n\n{format_prompt_list(agenda_rules)}\n\n" if agenda_rules else ""


def agent_has_tools(agent: Agent) -> bool:
    """Check if an agent has any tools available.

    :param agent: The agent to check.
    :return: True if the agent has tools, False otherwise.
    """
    return len(agent.available_tools) > 0


def format_references(
    references: tuple[str, ...], reference_type: str, intro: str
) -> str:
    """Formats references (e.g., contexts, summaries) for the prompt.

    :param references: The references.
    :param reference_type: The type of the references (e.g., "context", "summary").
    :param intro: The introduction to the references.
    :return: The formatted references.
    """
    if not references:
        return ""

    formatted_references = [
        f"[begin {reference_type} {reference_index + 1}]\n\n{reference}\n\n[end {reference_type} {reference_index + 1}]"
        for reference_index, reference in enumerate(references)
    ]

    joined_references = "\n\n".join(formatted_references)
    return f"{intro}\n\n{joined_references}\n\n"


def format_workflow_instruction(instruction: str) -> str:
    """Formats workflow instruction describing this phase's role in a sequential workflow.

    :param instruction: The workflow instruction text describing phase context and dependencies.
    :return: The formatted instruction, or empty string if no instruction provided.
    """
    if not instruction:
        return ""
    return f"**Workflow Context:**\n\n{instruction}\n\n"


# Team meeting prompts
def team_meeting_start_prompt(
    team_lead: Agent,
    team_members: tuple[Agent, ...],
    agenda: str,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    num_rounds: int = 1,
    workflow_instruction: str = "",
) -> str:
    """Generates the start prompt for a tean meeting.

    :param team_lead: The team lead.
    :param team_members: The team members.
    :param agenda: The agenda for the meeting.
    :param agenda_questions: The agenda questions to answer by the end of the meeting.
    :param agenda_rules: The rules for the agenda.
    :param summaries: The summaries of previous meetings.
    :param contexts: The contexts for the meeting.
    :param num_rounds: The number of rounds of discussion.
    :param workflow_instruction: Optional instruction describing this phase's role in a sequential workflow.
    :return: The start prompt for the tean meeting.
    """
    return (
        f"This is the beginning of a team meeting to discuss your research project. "
        f"This is a meeting with the team lead, {team_lead.title}, and the following team members: "
        f"{', '.join(team_member.title for team_member in team_members)}.\n\n"
        f"{format_workflow_instruction(workflow_instruction)}"
        f"{format_references(contexts, reference_type='context', intro='Here is context for this meeting:')}"
        f"{format_references(summaries, reference_type='summary', intro='Here are summaries of the previous meetings:')}"
        f"{format_agenda(agenda)}"
        f"{format_agenda_questions(agenda_questions)}"
        f"{format_agenda_rules(agenda_rules)}"
        f"{team_lead} will convene the meeting. "
        f"Then, each team member will provide their thoughts on the discussion one-by-one in the order above. "
        f"After all team members have given their input, {team_lead} will {SYNTHESIS_PROMPT}. "
        f"This will continue for {num_rounds} rounds. Once the discussion is complete, {team_lead} will {SUMMARY_PROMPT}."
    )


def team_meeting_team_lead_initial_prompt(team_lead: Agent) -> str:
    """Generates the initial prompt for the team lead in a team meeting.

    :param team_lead: The team lead.
    :return: The initial prompt for the team lead.
    """
    return (
        f"{team_lead}, please provide your initial thoughts on the agenda and any questions "
        "to guide the team discussion."
    )


def team_meeting_team_member_prompt(
    team_member: Agent,
    round_num: int,
    num_rounds: int,
    verbosity: "VerbosityConfig | None" = None,
    message_history: list = None,
) -> str:
    """Generates the prompt for a team member in a team meeting.

    :param team_member: The team member.
    :param round_num: The current round number.
    :param num_rounds: The total number of rounds.
    :param verbosity: Verbosity configuration. If None, uses VERBOSE (current behavior).
    :return: The prompt for the team member.
    """
    # Import here to avoid circular imports
    from veritas.verbosity import get_verbosity_config, VerbosityLevel

    if verbosity is None:
        verbosity = get_verbosity_config(VerbosityLevel.VERBOSE)

    # Check if this agent previously executed code in this meeting
    has_previous_code_execution = False
    if message_history:
        for msg in message_history:
            msg_content = msg.content if hasattr(msg, 'content') else str(msg)
            if ("✅ Code Executed Successfully" in msg_content or
                "❌ Code Execution Failed" in msg_content):
                # Check if it was this agent's code
                if hasattr(msg, 'name') and msg.name == team_member.title:
                    has_previous_code_execution = True
                    break
                # Or check content for session/filename hints
                if team_member.title.lower() in msg_content.lower():
                    has_previous_code_execution = True
                    break

    # MINIMAL: Just the agent name and a brief instruction
    if verbosity.level == VerbosityLevel.MINIMAL:
        prompt = f"{team_member}, your thoughts on the discussion?"
        if has_previous_code_execution:
            prompt += " (Note: You already ran code earlier - discuss those results.)"
        return prompt

    # Build prompt based on verbosity settings
    parts = [f"{team_member}, please provide your thoughts on the discussion"]

    if verbosity.include_round_counter:
        parts[0] += f" (round {round_num} of {num_rounds})"
    parts[0] += "."

    if verbosity.include_expertise_reminder:
        parts.append(
            f"Based on your expertise in {team_member.expertise}, "
            "analyze the agenda and provide your expert perspective."
        )

    if verbosity.include_goal_reminder:
        parts.append(f"Your goal: {team_member.goal}")

    # Standard additions (always include in STANDARD and VERBOSE)
    parts.append('If you do not have anything new or relevant to add, you may say "pass".')
    parts.append(
        "Remember that you can and should (politely) disagree with other team members "
        "if you have a different perspective."
    )

    base_prompt = " ".join(parts)

    # Add tool usage emphasis for agents with tools (STANDARD and VERBOSE only)
    if agent_has_tools(team_member) and verbosity.include_expertise_reminder:
        base_prompt += (
            "\n\nYou have tools available - use them to complete your tasks. "
            "Call the appropriate tools rather than just describing what you would do."
        )

    return base_prompt


def team_meeting_team_lead_intermediate_prompt(
    team_lead: Agent, round_num: int, num_rounds: int
) -> str:
    """Generates the intermediate prompt for the team lead in a team meeting at the end of a round of discussion.

    :param team_lead: The team lead.
    :param round_num: The current round number.
    :param num_rounds: The total number of rounds.
    :return: The intermediate prompt for the team lead.
    """
    return f"This concludes round {round_num} of {num_rounds} of discussion. {team_lead}, please {SYNTHESIS_PROMPT}."


def team_meeting_team_lead_final_prompt(
    team_lead: Agent,
    agenda: str,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    verbosity: "VerbosityConfig | None" = None,
    summary_instructions: str = "",
) -> str:
    """Generates the final prompt for the team lead in a team meeting to summarize the discussion.

    :param team_lead: The team lead.
    :param agenda: The agenda for the meeting.
    :param agenda_questions: The agenda questions to answer by the end of the meeting.
    :param agenda_rules: The rules for the agenda.
    :param verbosity: Verbosity configuration. If None, uses VERBOSE (current behavior).
    :param summary_instructions: Custom instructions for the summary (e.g., JSON output format).
    :return: The final prompt for the team lead.
    """
    # Import here to avoid circular imports
    from veritas.verbosity import get_verbosity_config, VerbosityLevel

    if verbosity is None:
        verbosity = get_verbosity_config(VerbosityLevel.VERBOSE)

    # Base prompt always includes the summary request
    prompt = f"{team_lead}, please {SUMMARY_PROMPT}.\n\n"

    # MINIMAL and STANDARD: Reference to agenda in history, no re-injection
    # VERBOSE: Full re-injection of agenda, questions, rules
    if verbosity.reinject_agenda:
        prompt += format_agenda(agenda, intro='As a reminder, here is the agenda for the meeting:')

    if verbosity.reinject_questions_rules:
        prompt += format_agenda_questions(
            agenda_questions,
            intro='As a reminder, here are the agenda questions that must be answered:'
        )
        prompt += format_agenda_rules(
            agenda_rules,
            intro='As a reminder, here are the agenda rules that must be followed:'
        )

    # Always include summary structure guidance
    prompt += "Your summary should take the following form.\n\n"
    prompt += summary_structure_prompt(has_agenda_questions=len(agenda_questions) > 0)

    # Add custom summary instructions if provided
    if summary_instructions:
        prompt += f"\n\n{summary_instructions}"

    return prompt


# Individual meeting prompts
def individual_meeting_start_prompt(
    team_member: Agent,
    agenda: str,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    workflow_instruction: str = "",
) -> str:
    """Generates the start prompt for an individual meeting.

    :param team_member: The team member.
    :param agenda: The agenda for the meeting.
    :param agenda_questions: The agenda questions to answer by the end of the meeting.
    :param agenda_rules: The rules for the agenda.
    :param summaries: The summaries of previous meetings.
    :param contexts: The contexts for the meeting.
    :param workflow_instruction: Optional instruction describing this phase's role in a sequential workflow.
    :return: The start prompt for the individual meeting.
    """
    return (
        f"This is the beginning of an individual meeting with {team_member} to discuss your research project.\n\n"
        f"{format_workflow_instruction(workflow_instruction)}"
        f"{format_references(contexts, reference_type='context', intro='Here is context for this meeting:')}"
        f"{format_references(summaries, reference_type='summary', intro='Here are summaries of the previous meetings:')}"
        f"{format_agenda(agenda)}"
        f"{format_agenda_questions(agenda_questions)}"
        f"{format_agenda_rules(agenda_rules)}"
        f"{team_member}, please provide your response to the agenda."
    )


def individual_meeting_critic_prompt(
    critic: Agent,
    agent: Agent,
) -> str:
    """Generates the intermediate prompt for the critic in an individual meeting.

    :param critic: The critic.
    :param agent: The agent that the critic is criticizing.
    """
    # Check if this is a code-as-output agent (Coding prefix in title)
    is_coding_agent = "Coding" in agent.title or "coding" in agent.title.lower()

    base_prompt = (
        f"{critic.title}, please critique {agent.title}'s most recent answer. "
        "In your critique, suggest improvements that directly address the agenda and any agenda questions. "
        "Prioritize simple solutions over unnecessarily complex ones, but demand more detail where detail is lacking. "
        "Additionally, validate whether the answer strictly adheres to the agenda and any agenda questions and provide corrective feedback if it does not. "
        "Only provide feedback; do not implement the answer yourself."
    )

    if is_coding_agent:
        # Add code-specific guidance for critics reviewing coding agents
        code_guidance = (
            "\n\n**Code Execution Context:**\n"
            "- Check message history for code execution feedback (✅ or ❌ markers)\n"
            "- If code was executed successfully, focus on output quality and completeness\n"
            "- If code failed, suggest specific fixes for the error\n"
            "- If agent wrote code WITHOUT triple backticks (```python), this is CRITICAL - code won't execute as plain text. Demand proper formatting.\n"
            "- If all required outputs exist and code succeeded, DO NOT ask for more code - confirm completion and request a plain-text summary (no code blocks)\n"
            "- Do NOT request extra scripts/reports beyond what the agenda requires\n"
            "- Flag inappropriate practices (e.g., synthetic data via np.random/random or hardcoded arrays used for results/plots) unless explicitly requested\n"
            "- Ensure effect size labeling matches the test used (e.g., Cohen's d for t-test, rank-biserial for Mann-Whitney)\n"
            "- Only request covariate adjustment if the plan/hypothesis explicitly specifies predictors; otherwise treat as optional limitation\n"
            "- If exact results exist (e.g., statistical_results.json), require the summary to cite exact values (avoid vague '<0.05' when precise numbers are available)\n"
            "- Only request re-execution if there's an actual error or missing required output"
        )
        return base_prompt + code_guidance

    return base_prompt


def team_meeting_critic_prompt(
    critic: Agent,
    team_lead: Agent,
) -> str:
    """Generates the prompt for the critic to review team discussion.

    :param critic: The critic agent.
    :param team_lead: The team lead agent.
    """
    return (
        f"{critic.title}, please review the team's discussion in this round. "
        f"Provide constructive feedback on the quality of the discussion, the depth of analysis, and how well the team is addressing the agenda. "
        "Highlight any gaps in reasoning, missing perspectives, or areas where the team could improve. "
        "Be specific and actionable in your feedback, but do not solve the problem yourself - the team should address your concerns in the next round."
    )


def individual_meeting_agent_prompt(
    critic: Agent,
    agent: Agent,
    is_final_round: bool = False,
    summary_instructions: str = "",
) -> str:
    """Generates the intermediate prompt for the agent in an individual meeting.

    :param critic: The critic.
    :param agent: The agent.
    :param is_final_round: Whether this is the final round (triggers structured output guidance).
    :param summary_instructions: Optional final-output instructions/checklist appended in the final round.
    """
    # Check if this is a code-as-output agent
    is_coding_agent = "Coding" in agent.title or "coding" in agent.title.lower()

    base_prompt = f"{agent.title}, please address {critic.title}'s most recent feedback. "

    if is_coding_agent:
        base_prompt += (
            "If you've already executed code successfully and produced all required outputs, "
            "focus on clarifying your explanation or providing a summary in plain text (no code blocks). "
            "Only write new code if the critic identified an actual error, missing output, or incorrect result. "
            "Do NOT rewrite working code just for style improvements."
        )
    else:
        base_prompt += (
            "If you've already completed the core task successfully, focus on clarifying or improving your explanation. "
            "Only re-execute tools if the critic identified an actual error or missing work that requires it."
        )

    if is_final_round:
        base_prompt += (
            "\n\nThis is your FINAL response. Please structure it with these sections:\n\n"
            "### Summary\n"
            "Overview of what was accomplished.\n\n"
            "### Key Findings\n"
            "Main results with specific numbers, metrics, and outcomes.\n\n"
            "### Recommendation\n"
            "Your expert recommendation.\n\n"
            "### Next Steps\n"
            "What should happen next."
        )
        if summary_instructions:
            base_prompt += f"\n\n{summary_instructions.strip()}"

    return base_prompt


CODING_RULES = (
    "Your code must be self-contained (with appropriate imports) and complete.",
    "Your code may not include any undefined or unimplemented variables or functions.",
    "Your code may not include any pseudocode; it must be fully functioning code.",
    "Your code may not include any hard-coded examples.",
    "If your code needs user-provided values, write code to parse those values from the command line.",
    "Your code must be high quality, well-engineered, efficient, and well-documented (including docstrings, comments, and Python type hints if using Python).",
)
