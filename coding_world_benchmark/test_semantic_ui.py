"""The UI and API wiring for the semantic layers.

Source-level assertions, following the convention
``test_electron_ui_workspace_threads.py`` set: the React bundle and the desktop
app cannot be imported on a machine without a browser toolchain or tkinter, and
the thing worth pinning is that the panel and the endpoints stay connected to the
session rather than drifting into a second copy of the logic.
"""
from pathlib import Path

ROOT = Path(__file__).parent
MAIN = ROOT / "web" / "src" / "main.jsx"
STYLES = ROOT / "web" / "src" / "styles.css"
BRIDGE = ROOT / "coding_agent_bridge.py"
APP = ROOT / "coding_agent_app.py"
SERVICE = ROOT / "semantic_service.py"


def test_the_panel_is_a_third_inspector_tab():
    source = MAIN.read_text(encoding="utf-8")

    assert "function SemanticPanel({ state, templates, theorem, onTheorem, onAction, onCompose, onSend })" in source
    assert "setActiveTab('semantics')" in source
    assert "<SemanticPanel state={semantic} templates={templates}" in source


def test_the_panel_reads_the_session_rather_than_the_last_message():
    source = MAIN.read_text(encoding="utf-8")

    assert "const refreshSemantic = () =>" in source
    assert "`${API_BASE}/semantic/state`" in source
    assert "if (data.semantic) setSemantic(data.semantic)" in source
    assert "else refreshSemantic()" in source


def test_every_panel_control_is_an_action_the_layers_define():
    source = MAIN.read_text(encoding="utf-8")

    # L8.33: answering the pending question, including the reply that commits
    # nothing.
    assert "onAction('answer', { text: 'はい' })" in source
    assert "onAction('answer', { text: 'いいえ' })" in source
    assert "onAction('answer', { text: '分からない' })" in source
    # L8.42: a blocked question is unblocked by opening the episode, not by
    # answering it anyway.
    assert "state.blocked &&" in source
    assert "onAction('open', { episode: state.episode })" in source
    # L8.36: naming the model when time and episode both explain the evidence.
    assert "state.structural_question &&" in source
    assert "onAction('model', { label: model })" in source


def test_the_templates_are_editable_and_not_fixed_buttons():
    source = MAIN.read_text(encoding="utf-8")

    assert "function TemplatePicker({ templates, onCompose, onAction, onSend })" in source
    # A select for the form, inputs for its slots, and a textarea for the text
    # itself -- a template nobody can change is a button.
    assert 'className="template-select"' in source
    assert 'className="template-slot"' in source
    assert 'className={`template-draft ${asCode ? \'code\' : \'\'}`}' in source
    assert "onChange={(event) => setDraft(event.target.value)}" in source
    assert "onChange={(event) => editSlot(slot.name, event.target.value)}" in source


def test_a_template_is_sent_by_the_route_its_kind_belongs_to():
    source = MAIN.read_text(encoding="utf-8")

    assert "if (template.kind === 'REQUEST') onAction('ask', { message: text }, text)" in source
    assert "else if (template.kind === 'ANSWER') onAction('answer', { text }, text)" in source
    assert "else onSend(text)" in source
    # What the person sent belongs in the transcript, not only its reply.
    assert "const semanticAction = (path, body, echo) =>" in source


def test_the_template_catalogue_is_fetched_from_the_bridge():
    source = MAIN.read_text(encoding="utf-8")

    assert "`${API_BASE}/semantic/templates`" in source
    assert "setTemplates(data.templates)" in source
    assert "<TemplatePicker templates={templates}" in source


def test_the_panel_says_where_the_templates_come_from():
    source = MAIN.read_text(encoding="utf-8")

    assert "VERIFIED TEMPLATES" in source
    assert "契約検証が毎回そのまま再生して" in source
    assert "{template.verified_by}" in source


def test_the_templates_have_styles():
    css = STYLES.read_text(encoding="utf-8")

    for selector in (".template-select", ".template-slot", ".template-draft",
                     ".template-meta", ".template-note"):
        assert selector in css


def test_the_script_editor_is_a_panel_section():
    source = MAIN.read_text(encoding="utf-8")

    assert "function ScriptEditor({ script, onAction, onSend })" in source
    assert "<ScriptEditor script={state.script}" in source
    assert 'className="script-source"' in source
    assert "onAction('script', { source: text }, text)" in source
    # The refusal is shown as a refusal, not as an empty result.
    assert "script.ok" in source and 'className="script-error"' in source


def test_the_editor_offers_both_languages():
    source = MAIN.read_text(encoding="utf-8")

    assert "const EXAMPLE_SCRIPTS = {" in source
    assert "lorentz: F exists" in source
    assert ">対話の例<" in source and ">物理の例<" in source


def test_unbound_symbols_are_shown_rather_than_guessed():
    source = MAIN.read_text(encoding="utf-8")

    assert "script.needs && script.needs.length > 0" in source
    assert "値が決まっていない記号" in source
    assert "推測しないので実行していません" in source
    assert ".script-needs" in STYLES.read_text(encoding="utf-8")


def test_the_theorem_bundle_has_its_own_panel_section():
    source = MAIN.read_text(encoding="utf-8")

    assert "function TheoremPanel({ status, onTheorem })" in source
    assert "<TheoremPanel status={theorem} onTheorem={onTheorem} />" in source
    assert "`${API_BASE}/theorem/status`" in source
    for action in ("search", "run", "obligations", "verify"):
        assert f"onTheorem('{action}'" in source


def test_the_two_dispatchers_are_named_apart():
    """A theorem action must not be readable as a semantic one."""
    source = MAIN.read_text(encoding="utf-8")

    assert "const theoremAction = (path, body, echo) =>" in source
    assert "`${API_BASE}/theorem/${path}`" in source
    start = source.index("function TheoremPanel(")
    end = source.index("// Two languages, one box.")
    assert "onAction(" not in source[start:end]


def test_the_panel_keeps_the_sieve_and_the_proof_apart():
    source = MAIN.read_text(encoding="utf-8")

    assert "篩を通った本数（asserted）と証明された本数（valid）は別に出ます" in source
    assert "Lean が呼ばれていなければ、証明は 0 です" in source


def test_the_panel_says_the_script_is_not_evaluated():
    source = MAIN.read_text(encoding="utf-8")

    assert "評価はされません" in source
    assert "1行でも読めなければ1行も実行しません" in source


def test_a_template_can_be_shown_as_prose_or_as_a_call():
    source = MAIN.read_text(encoding="utf-8")

    assert "const [asCode, setAsCode] = useState(false)" in source
    assert "const toggleCode = (code) =>" in source
    assert ">文として<" in source and ">コードとして<" in source
    assert "if (asCode) onAction('script', { source: text }, text)" in source
    # A template with no code form cannot be switched into one.
    assert "disabled={!template.code}" in source


def test_the_script_styles_exist():
    css = STYLES.read_text(encoding="utf-8")

    for selector in (".script-source", ".script-result", ".script-row",
                     ".script-error", ".template-modes"):
        assert selector in css


def test_the_panel_labels_the_invariants_as_invariants():
    source = MAIN.read_text(encoding="utf-8")

    assert "SESSION INVARIANTS" in source
    assert "真値を必要としない" in source
    assert "invariant broken" not in source  # the class is composed, not hard-coded
    assert "className={`context-row invariant ${value ? 'broken' : ''}`}" in source


def test_the_panel_has_styles_for_every_state_it_reports():
    css = STYLES.read_text(encoding="utf-8")

    for selector in (".semantic-status", ".semantic-status.resolved",
                     ".semantic-status.blocked", ".semantic-status.impossible",
                     ".semantic-question", ".semantic-question.structural",
                     ".semantic-blocked", ".episode-row", ".access-dot.open",
                     ".access-dot.known", ".context-row.invariant.broken"):
        assert selector in css


def test_the_bridge_exposes_the_session_and_its_actions():
    source = BRIDGE.read_text(encoding="utf-8")

    assert "semantics = SemanticSession()" in source
    assert '"/api/semantic/state"' in source
    for endpoint in ("ask", "answer", "open", "episode", "model", "reset", "rerun"):
        assert f'"/api/semantic/{endpoint}"' in source
    assert '"/api/semantic/templates"' in source
    assert '"/api/semantic/script"' in source
    for endpoint in ("status", "search", "run", "verify", "obligations"):
        assert f'"/api/theorem/{endpoint}"' in source


def test_a_pending_question_owns_the_next_utterance_on_both_surfaces():
    """The reply is only interpretable against the question it answers."""
    for source in (BRIDGE.read_text(encoding="utf-8"),
                   APP.read_text(encoding="utf-8")):
        assert "handle_command(self" in source or "handle_command(" in source
        assert "dialogue.pending is not None" in source


def test_the_feature_list_advertises_the_semantic_commands():
    assert "SEMANTIC_FEATURE_COMMANDS" in BRIDGE.read_text(encoding="utf-8")
    assert "+ SEMANTIC_FEATURE_COMMANDS" in APP.read_text(encoding="utf-8")


def test_the_service_does_not_claim_an_accuracy_it_cannot_measure():
    source = SERVICE.read_text(encoding="utf-8")

    # L8.40 computes wrong_answer against a known truth. A live session has no
    # oracle, so the module says so instead of producing the number anyway.
    assert "there is no oracle" in source
    assert "executed_without_authority" in source
    assert "authority_while_disputed" in source
    assert "committed_without_commitment" in source
    assert "stale_episode_leak" in source


def test_the_service_needs_no_desktop_toolkit():
    source = SERVICE.read_text(encoding="utf-8")

    assert "tkinter" not in source
    assert "import tkinter" not in source
