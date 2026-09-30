from coding_world_benchmark.coding_agent_app import LocalWorkspaceAgent
from coding_world_benchmark.structured_repository_report_l62_experiment import RequestType, classify_request
from coding_world_benchmark.workspace_operations_l75_experiment import (
    WorkspaceAction,
    WorkspaceExecutor,
    execute_delete_operation,
    execute_move_operation,
    format_operation_report,
    is_delete_directory_request,
    is_move_directories_request,
    parse_delete_operation,
    parse_move_operation,
)


def test_delete_request_routes_to_explicit_action():
    request = "SU2, SU3の検証フォルダを削除して"
    assert is_delete_directory_request(request)
    assert classify_request(request).request_type is RequestType.DELETE_DIRECTORY
    assert parse_delete_operation(request).action is WorkspaceAction.DELETE_DIRECTORY


def test_delete_moves_unique_directories_to_trash_and_verifies(tmp_path):
    su2 = tmp_path / "su2_validation"
    su3 = tmp_path / "su3_validation"
    su2.mkdir()
    su3.mkdir()
    (su2 / "result.txt").write_text("x", encoding="utf-8")

    operation = parse_delete_operation("SU2, SU3の検証フォルダを削除して")
    results = execute_delete_operation(tmp_path, operation)

    assert all(result.success for result in results)
    assert not su2.exists()
    assert not su3.exists()
    assert all(result.trash_path and "agent_trash" in result.trash_path for result in results)


def test_ambiguous_delete_does_not_move_any_directory(tmp_path):
    first = tmp_path / "su2_validation_a"
    second = tmp_path / "su2_validation_b"
    first.mkdir()
    second.mkdir()

    results = execute_delete_operation(tmp_path, parse_delete_operation("SU2の検証フォルダを削除して"))

    assert results[0].status == "AMBIGUOUS_TARGET"
    assert first.exists() and second.exists()


def test_path_escape_is_rejected(tmp_path):
    executor = WorkspaceExecutor(tmp_path)
    try:
        executor.resolver.resolve_safe("../../Windows")
    except PermissionError as error:
        assert "escapes workspace" in str(error)
    else:
        raise AssertionError("workspace escape was not rejected")


def test_app_deletes_instead_of_returning_repo_search(tmp_path):
    target = tmp_path / "su2_validation"
    target.mkdir()
    result = LocalWorkspaceAgent(tmp_path).handle("SU2の検証フォルダを削除して")
    assert "Workspace Operation Report" in result.text
    assert "TRASHED" in result.text
    assert not target.exists()


def test_move_request_has_explicit_action_and_destination():
    request = "SU2, SU3の検証フォルダを一つの新規フォルダにまとめて移動させて"
    assert is_move_directories_request(request)
    operation = parse_move_operation(request)
    assert operation.action is WorkspaceAction.MOVE_DIRECTORIES
    assert operation.destination_name == "su2_su3"
    assert len(operation.target_descriptions) == 2
    assert classify_request(request).request_type is RequestType.MOVE_DIRECTORIES


def test_move_creates_parent_moves_both_directories_and_verifies(tmp_path):
    su2 = tmp_path / "su2_ordered_loop_experiment"
    su3 = tmp_path / "su3_tuned_wilson_critical_benchmark"
    su2.mkdir()
    su3.mkdir()
    (su2 / "result.txt").write_text("su2", encoding="utf-8")
    (su3 / "result.txt").write_text("su3", encoding="utf-8")

    operation = parse_move_operation("SU2, SU3の検証フォルダを一つの新規フォルダにまとめて移動させて")
    results = execute_move_operation(tmp_path, operation)
    destination = tmp_path / "su2_su3"

    assert all(result.success for result in results)
    assert not su2.exists() and not su3.exists()
    assert (destination / su2.name / "result.txt").read_text(encoding="utf-8") == "su2"
    assert (destination / su3.name / "result.txt").read_text(encoding="utf-8") == "su3"


def test_move_all_matching_groups_and_explicit_destination(tmp_path):
    directories = (
        "su2_ordered_loop_experiment",
        "su2_wilson_validation",
        "su3_tuned_wilson_critical_benchmark",
        "su3_noncommutative_target_experiment",
    )
    for name in directories:
        (tmp_path / name).mkdir()

    operation = parse_move_operation("SU2, SU3の検証フォルダを新規フォルダsu2_su3を作成してここにまとめて移動させて")
    results = execute_move_operation(tmp_path, operation)
    destination = tmp_path / "su2_su3"

    assert operation.target_descriptions == ("SU2 validation", "SU3 validation")
    assert operation.target_mode == "all_matching"
    assert operation.destination_name == "su2_su3"
    assert len(results) == 4
    assert all(result.success for result in results)
    assert all(not (tmp_path / name).exists() for name in directories)
    assert all((destination / name).is_dir() for name in directories)


def test_failed_group_resolution_does_not_create_destination(tmp_path):
    (tmp_path / "su2_validation_a").mkdir()
    (tmp_path / "su2_validation_b").mkdir()

    operation = parse_move_operation("SU2の検証フォルダを新規フォルダsu2_su3を作成して移動させて")
    results = execute_move_operation(tmp_path, operation)

    assert results[0].status == "AMBIGUOUS_TARGET"
    assert not (tmp_path / "su2_su3").exists()
    assert "実際の移動は行われていません" in format_operation_report(operation, results)


def test_app_move_precedes_validation_memory(tmp_path):
    su2 = tmp_path / "su2_validation"
    su3 = tmp_path / "su3_validation"
    su2.mkdir()
    su3.mkdir()
    agent = LocalWorkspaceAgent(tmp_path)
    agent.validation_knowledge = (object(),)

    result = agent.handle("SU2, SU3の検証フォルダを一つの新規フォルダにまとめて移動させて")

    assert "MOVE_DIRECTORIES" in result.text
    assert "Integrated" not in result.text
    assert not su2.exists() and not su3.exists()
