from pathlib import Path


ROOT = Path(__file__).parent
MAIN = ROOT / "web" / "src" / "main.jsx"
STYLES = ROOT / "web" / "src" / "styles.css"


def test_workspace_activation_loads_threads_for_health_and_chat_workspace_changes():
    source = MAIN.read_text(encoding="utf-8")

    assert "const activateWorkspace = (path, syncBridge = false)" in source
    assert "if (data.workspace) activateWorkspace(data.workspace, false)" in source
    assert "if (data.workspace && data.workspace !== workspace) activateWorkspace(data.workspace, false)" in source
    assert "loadWorkspaceThreads(path)" in source
    assert "rememberWorkspace(path)" in source


def test_threads_have_context_menu_delete_command():
    source = MAIN.read_text(encoding="utf-8")

    assert "const deleteThread = (threadId)" in source
    assert "onContextMenu={(event) => { event.preventDefault(); setThreadMenu" in source
    assert "Delete thread" in source
    assert "deleteThread(threadMenu.threadId)" in source
    assert "store[workspace] = next" in source


def test_left_sidebar_remains_fixed_and_thread_list_scrolls():
    css = STYLES.read_text(encoding="utf-8")

    assert "body { margin: 0; min-width: 920px; overflow: hidden; }" in css
    assert ".app-shell { height: 100vh;" in css
    assert ".sidebar { position: sticky; top: 0; height: 100vh;" in css
    assert ".thread-panel { flex: 1 1 160px;" in css
    assert ".thread-list { min-height: 0; overflow-y: auto;" in css
    assert ".thread-context-menu { position: fixed;" in css
