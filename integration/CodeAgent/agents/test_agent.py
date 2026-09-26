"""
 Agent - 
"""
import json
import os
import subprocess
from pathlib import Path
from pathlib import Path as _Path
from typing import Optional, Dict

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from llm_client import get_openai_client


class TestResult(BaseModel):
    """"""
    passed: bool = Field(description="")
    test_output: str = Field(description="")
    failed_test: str = Field(description="")
    error_message: Optional[str] = Field(default=None, description="")


class TestReflection(BaseModel):
    """"""
    issues: list = Field(default_factory=list, description="")
    suggestions: list = Field(default_factory=list, description="")


class TestAgent:
    """DevEval  Agent（）"""

    def __init__(
        self,
        conda_env: str = "DevEval",
        script_path: Path = Path(""),
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        **kwargs
    ):
        self.conda_env = conda_env
        self.script_path = script_path
        self.llm = get_openai_client(model=model_name, temperature=temperature, **kwargs)
        self.test_history = []

    def test(
        self,
        namespace: str,
        generated_code: str,
        tests: list,
        project_path: str,
        sample_id: Optional[str] = None,
    ) -> TestResult:
        """
        DevEval ： check_source_code.py ， conda  pass_k_agent.py
        sample_id ，
        """
        check_script = Path("")
        source_code_dir = Path("")

        try:
            subprocess.run(
                ["bash", "-c", "find . -name '.eggs' -type d -exec rm -rf {} +"],
                cwd=str(source_code_dir / project_path),
                check=False,
            )
        except Exception as cleanup_err:
            print(f"[TestAgent] .eggs cleanup failed: {cleanup_err}")

        try:
            print(f"[TestAgent] Running check_source_code_with_repo.py on {source_code_dir} / {project_path}")
            check_result = subprocess.run(
                ["python", str(check_script), str(source_code_dir), str(project_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if check_result.returncode != 0:
                return TestResult(
                    passed=False,
                    test_output=check_result.stdout,
                    failed_test="",
                    error_message=f"Source code check failed:\n{check_result.stderr}",
                )
            print("[TestAgent] Source code check passed.")
            print(check_result.stdout)
        except subprocess.TimeoutExpired as te:
            return TestResult(
                passed=False,
                test_output=te.stdout or "",
                failed_test="",
                error_message=f"TimeoutExpired during source code check: {str(te)}",
            )
        except Exception as e:
            return TestResult(
                passed=False,
                test_output="",
                failed_test="",
                error_message=f"Exception during source code check: {str(e)}",
            )

        command = [
            "conda", "run", "--name", self.conda_env, "python",
            str(self.script_path), "--namespace", namespace, "--completion", generated_code,
        ]
        project_cwd = _Path("")

        output = ""
        error = ""
        passed = False
        failed_test_func = ""

        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_cwd) + os.pathsep + env.get("PYTHONPATH", "")
            print("[TestAgent] PYTHONPATH:", env["PYTHONPATH"])

            result = subprocess.run(
                command,
                cwd=str(project_cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )

            parts = result.stdout.split("&&", 2)
            output = parts[0] if len(parts) > 0 else ""
            failed_test = parts[1].split("::") if len(parts) > 1 else []
            if len(parts) < 3:
                error = f"Raw Output (Format Mismatch):\n{result.stdout}\nSTDERR:\n{result.stderr}"
            else:
                error = parts[2]

            passed = (result.returncode == 0) and ("Pass" in output)
            test_class = failed_test[1] if len(failed_test) > 2 else ""
            test_name = failed_test[2] if len(failed_test) > 2 else (failed_test[1] if len(failed_test) > 1 else "")
            for item in tests:
                if item["class"] != test_class:
                    continue
                if item["name"] != test_name:
                    continue
                if "/".join(item["src_file"].split("/")[-2:]) != failed_test[0]:
                    continue
                failed_test_func = item["code_chunk"]
                break

        except subprocess.TimeoutExpired as te:
            passed = False
            output = te.stdout or ""
            error = f"TimeoutExpired: {str(te)}"
        except Exception as e:
            passed = False
            error = str(e)
            print(f"[TestAgent] Exception during test execution: {error}")

        return TestResult(
            passed=passed,
            test_output=output,
            failed_test=failed_test_func,
            error_message=error,
        )


class CoderEvalPythonTestAgent:
    """CoderEvalPython  Agent（ PythonExec.py）"""

    def __init__(
        self,
        container_name: str = "codereval",
        script_path: str = "",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout: int = 180,
        **kwargs,
    ):
        self.container_name = container_name
        self.script_path = script_path
        self.timeout = timeout
        self.llm = get_openai_client(model=model_name, temperature=temperature, **kwargs)
        self.test_history = []

    def _run_docker_cmd(self, args: list, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _ensure_container_running(self) -> Optional[str]:
        inspect = self._run_docker_cmd(
            ["inspect", "-f", "{{.State.Status}}", self.container_name],
            timeout=20,
        )
        if inspect.returncode != 0:
            msg = (inspect.stderr or inspect.stdout or "").strip()
            return (
                f"Cannot inspect container '{self.container_name}'. "
                f"Please ensure Docker is running and the container exists. Details: {msg}"
            )

        status = (inspect.stdout or "").strip().lower()
        if status == "running":
            return None

        start = self._run_docker_cmd(["start", self.container_name], timeout=30)
        if start.returncode != 0:
            msg = (start.stderr or start.stdout or "").strip()
            return (
                f"Container '{self.container_name}' is '{status}' and failed to start. "
                f"Details: {msg}"
            )

        return None

    def test(
        self,
        namespace: str,
        generated_code: str,
        tests: list,
        project_path: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> TestResult:
        """
        CoderEvalPython ：
        docker exec codereval python  --id <_id> --code <generated_code>
         JSON  pass/error
        """
        if not sample_id:
            return TestResult(
                passed=False,
                test_output="",
                failed_test="",
                error_message="Missing sample_id for CoderEvalPython test.",
            )

        container_error = self._ensure_container_running()
        if container_error:
            return TestResult(
                passed=False,
                test_output="",
                failed_test="",
                error_message=container_error,
            )

        command = [
            "docker", "exec", self.container_name,
            "python", self.script_path,
            "--id", str(sample_id),
            "--code", generated_code,
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()

            parsed = {}
            if stdout:
                try:
                    parsed = json.loads(stdout.splitlines()[-1])
                except Exception:
                    parsed = {}

            passed = bool(parsed.get("pass", False))
            error_message = parsed.get("error")
            if error_message is None:
                if stderr:
                    error_message = stderr
                elif not passed:
                    error_message = f"Unexpected test output: {stdout}"
                else:
                    error_message = ""

            # Ensure reflection agent gets full raw failure output for CoderEvalPython.
            if not passed:
                raw_output = stdout or stderr
                if raw_output:
                    if error_message:
                        if raw_output not in error_message:
                            error_message = f"{error_message}\n\n[Raw Test Output]\n{raw_output}"
                    else:
                        error_message = raw_output

            return TestResult(
                passed=passed,
                test_output=stdout,
                failed_test="",
                error_message=error_message,
            )
        except subprocess.TimeoutExpired as te:
            return TestResult(
                passed=False,
                test_output=te.stdout or "",
                failed_test="",
                error_message=f"TimeoutExpired: {str(te)}",
            )
        except Exception as e:
            return TestResult(
                passed=False,
                test_output="",
                failed_test="",
                error_message=str(e),
            )


def get_test_agent(dataset: str, **kwargs):
    """ Agent； DevEval """
    if (dataset or "").strip().lower() == "coderevalpython":
        return CoderEvalPythonTestAgent(**kwargs)
    return TestAgent(**kwargs)
