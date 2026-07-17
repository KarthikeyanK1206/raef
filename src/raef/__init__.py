"""Top-level package for the Reliable Agent Execution Framework (RAEF).

This package contains the transaction, logging, recovery, and verification
modules for fault-tolerant tool execution.
"""

from .adapters import (
    ToolPolicy,
    build_execution_id,
    build_message_id,
    ensure_run_started,
    with_logging_service,
    wrap_langchain_tool,
)
from .logging_service import LoggingService
from .recovery.common import RecoveryAction, RecoveryDecision, WriteVerifierProtocol
from .recovery.recovery.check_helper import CheckRecoveryResult, recover_with_check_api
from .recovery.recovery.handler import RecoveryCoordinator
from .recovery.recovery.strategy import RuntimeRecoveryStrategy
from .evaluation import EvaluationRecorder, EvaluationSpanHandle, build_evaluation_report
from .verifier import MockTargetVerifier, VerificationDecision, verify_mock_target_write
from .txn_manager import (
    AmbiguousToolError,
    FunctionToolAdapter,
    ToolAdapterProtocol,
    TransactionDisposition,
    TransactionManager,
    TransactionResult,
)

__all__ = [
    "AmbiguousToolError",
    "FunctionToolAdapter",
    "ToolAdapterProtocol",
    "TransactionDisposition",
    "TransactionManager",
    "TransactionResult",
    "LoggingService",
    "CheckRecoveryResult",
    "RecoveryAction",
    "RecoveryCoordinator",
    "RecoveryDecision",
    "RuntimeRecoveryStrategy",
    "WriteVerifierProtocol",
    "EvaluationRecorder",
    "EvaluationSpanHandle",
    "build_evaluation_report",
    "MockTargetVerifier",
    "VerificationDecision",
    "verify_mock_target_write",
    "recover_with_check_api",
    "with_logging_service",
    "ensure_run_started",
    "ToolPolicy",
    "build_execution_id",
    "build_message_id",
    "wrap_langchain_tool",
]
