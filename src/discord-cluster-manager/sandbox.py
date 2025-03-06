import sys
import builtins
import importlib
import importlib.abc
import importlib.machinery
import types
import os
import re
import ast
import signal
import resource
import threading
from contextlib import contextmanager
from typing import Dict, List, Set, Optional, Any, Callable, Tuple

# List of dangerous functions and modules to restrict
RESTRICTED_MODULES = {
    'os': {'system', 'popen', 'spawn', 'exec', 'execl', 'execle', 'execlp', 'execlpe', 
           'execv', 'execve', 'execvp', 'execvpe', 'startfile', 'rename', 'remove', 'unlink',
           'rmdir', 'mkdir', 'makedirs', 'fork', 'forkpty', 'killpg', 'kill', '_exit', 'setuid',
           'seteuid', 'setreuid', 'setgid', 'setegid', 'setregid', 'chdir', 'fchdir', 'chroot',
           'chmod', 'chown', 'lchown', 'fchown', 'symlink', 'truncate', 'ftruncate', 'putenv',
           'unsetenv', 'environ'},
    'sys': {'exit', '_exit', 'modules', 'path', 'meta_path', 'exitfunc', 'displayhook'},
    'subprocess': {'*'},  # Block the entire module
    'multiprocessing': {'*'},  # Block the entire module
    'importlib': {'*'},  # Block direct importlib usage
    'builtins': {'__import__', 'eval', 'exec', 'compile', 'open', 'input', 'breakpoint'},
    'ctypes': {'*'},  # Block the entire module
    'shutil': {'copyfileobj', 'copyfile', 'copy', 'copy2', 'copytree', 'move', 'rmtree'},
    'socket': {'*'},  # Block network operations
    'urllib': {'*'},  # Block network operations
    'urllib.request': {'*'},  # Block network operations
    'http': {'*'},  # Block network operations
    'requests': {'*'},  # Block network operations
    'pip': {'*'},  # Block pip installations
    'setuptools': {'*'},  # Block package installations
    'pkg_resources': {'*'},  # Block package management
    'distutils': {'*'},  # Block package installations
}

# List of allowed imports for computation
ALLOWED_IMPORTS = {
    'torch', 'numpy', 'math', 'random', 'time', 'functools', 'itertools',
    'collections', 'copy', 'datetime', 'json', 're', 'typing',
    # Allow specific standard library modules that are safe
    'abc', 'array', 'bisect', 'calendar', 'contextlib', 'decimal', 'enum',
    'fractions', 'heapq', 'numbers', 'statistics', 'string', 'textwrap',
    # Allow submodules from torch
    'torch.nn', 'torch.optim', 'torch.cuda', 'torch.utils', 'torch.distributions',
    # Allow NumPy submodules
    'numpy.random', 'numpy.linalg', 'numpy.fft'
}

# Set to track currently allowed imports (can be dynamically modified)
_CURRENTLY_ALLOWED = set(ALLOWED_IMPORTS)

# Original built-in import function
_original_import = builtins.__import__

# Track if sandbox is active
_sandbox_active = False

class RestrictedImportError(ImportError):
    """Raised when an import is blocked by the sandbox."""
    pass

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    A replacement for the built-in __import__ function that restricts imports.
    """
    if not _sandbox_active:
        return _original_import(name, globals, locals, fromlist, level)

    # Check if the module is in the allowed list
    if name in _CURRENTLY_ALLOWED:
        return _original_import(name, globals, locals, fromlist, level)
    
    # Check if it's a submodule of an allowed module
    for allowed in _CURRENTLY_ALLOWED:
        if name.startswith(allowed + '.'):
            return _original_import(name, globals, locals, fromlist, level)
    
    # Explicit check for built-in modules that should always be allowed
    if name in sys.builtin_module_names and name not in RESTRICTED_MODULES:
        return _original_import(name, globals, locals, fromlist, level)
    
    # Check if the module is in the restricted list
    for restricted_module, restricted_attrs in RESTRICTED_MODULES.items():
        if name == restricted_module:
            if '*' in restricted_attrs:
                raise RestrictedImportError(
                    f"Import of module '{name}' is not allowed in the sandbox environment. "
                    f"This module is restricted for security reasons."
                )
            
            # If only specific attributes are restricted, wrap the module
            module = _original_import(name, globals, locals, fromlist, level)
            return _create_restricted_module(module, restricted_attrs)
    
    # For any other imports, log and deny
    raise RestrictedImportError(
        f"Import of module '{name}' is not allowed in the sandbox environment. "
        f"Only specific modules required for computational tasks are permitted."
    )

def _create_restricted_module(module, restricted_attrs):
    """Create a wrapper around a module that blocks access to specific attributes."""
    class RestrictedModule:
        def __init__(self, module, restricted_attrs):
            self._module = module
            self._restricted_attrs = restricted_attrs
            
        def __getattr__(self, name):
            if name in self._restricted_attrs:
                raise AttributeError(
                    f"Access to '{self._module.__name__}.{name}' is not allowed in the sandbox environment."
                )
            return getattr(self._module, name)
    
    return RestrictedModule(module, restricted_attrs)

def _restricted_exec(code, globals=None, locals=None):
    """
    A replacement for the built-in exec function that raises an error.
    """
    raise RuntimeError("Use of exec() is not allowed in the sandbox environment.")

def _restricted_eval(expr, globals=None, locals=None):
    """
    A replacement for the built-in eval function that raises an error.
    """
    raise RuntimeError("Use of eval() is not allowed in the sandbox environment.")

def _restricted_compile(*args, **kwargs):
    """
    A replacement for the built-in compile function that raises an error.
    """
    raise RuntimeError("Use of compile() is not allowed in the sandbox environment.")

def _restricted_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    """
    A replacement for the built-in open function that only allows reading from specific directories.
    """
    # Convert to absolute path
    if not os.path.isabs(file):
        file = os.path.abspath(file)
    
    # Only allow reading, not writing
    if 'w' in mode or 'a' in mode or '+' in mode or 'x' in mode:
        raise IOError(f"Write operations are not allowed in the sandbox environment.")
    
    # Call the original open function
    return _original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

# Original built-in functions
_original_open = builtins.open
_original_exec = builtins.exec
_original_eval = builtins.eval
_original_compile = builtins.compile

class Sandbox:
    """
    A context manager that creates a restricted execution environment for user code.
    """
    def __init__(self, additional_allowed_imports=None):
        self.additional_allowed_imports = additional_allowed_imports or []
        self.original_builtins = {}
        self.original_sys_modules = {}
    
    def __enter__(self):
        global _sandbox_active
        _sandbox_active = True
        
        # Add any additional allowed imports
        for module in self.additional_allowed_imports:
            _CURRENTLY_ALLOWED.add(module)
        
        # Save original built-ins and replace with restricted versions
        self.original_builtins = {
            '__import__': builtins.__import__,
            'exec': builtins.exec,
            'eval': builtins.eval,
            'compile': builtins.compile,
            'open': builtins.open,
        }
        
        # Replace built-ins with restricted versions
        builtins.__import__ = _safe_import
        builtins.exec = _restricted_exec
        builtins.eval = _restricted_eval
        builtins.compile = _restricted_compile
        builtins.open = _restricted_open
        
        # Disable sys.modules manipulation
        self.original_sys_modules = sys.modules.copy()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _sandbox_active
        _sandbox_active = False
        
        # Restore original built-ins
        for name, func in self.original_builtins.items():
            setattr(builtins, name, func)
        
        # Restore sys.modules
        for module_name in list(sys.modules.keys()):
            if module_name not in self.original_sys_modules:
                del sys.modules[module_name]
        
        # Remove any additional allowed imports we added
        for module in self.additional_allowed_imports:
            if module in _CURRENTLY_ALLOWED:
                _CURRENTLY_ALLOWED.remove(module)
        
        return False  # Don't suppress exceptions

# AST-based static code analyzer for submissions
class CodeAnalyzer(ast.NodeVisitor):
    """
    Static code analyzer that detects potentially malicious patterns in Python code.
    """
    def __init__(self):
        self.issues = []
        self.imports = set()
        self.suspicious_calls = []
        
    def visit_Import(self, node):
        for name in node.names:
            self.imports.add(name.name)
            # Check for suspicious imports
            if name.name in RESTRICTED_MODULES or name.name.split('.')[0] in RESTRICTED_MODULES:
                self.issues.append(f"Suspicious import: {name.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        module_name = node.module
        self.imports.add(module_name)
        # Check for suspicious imports
        if module_name in RESTRICTED_MODULES or module_name.split('.')[0] in RESTRICTED_MODULES:
            self.issues.append(f"Suspicious import from: {module_name}")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # Check for calls to exec, eval, etc.
        if isinstance(node.func, ast.Name):
            if node.func.id in {'exec', 'eval', 'compile', '__import__'}:
                self.issues.append(f"Suspicious call to {node.func.id}()")
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                # Check for os.system, subprocess.call, etc.
                if node.func.value.id in {'os', 'subprocess', 'sys', 'shutil'}:
                    self.suspicious_calls.append(f"{node.func.value.id}.{node.func.attr}")
                    self.issues.append(f"Suspicious call to {node.func.value.id}.{node.func.attr}()")
        self.generic_visit(node)

def analyze_code(code: str) -> Tuple[bool, List[str]]:
    """
    Performs static analysis on the code to detect potentially harmful patterns.
    
    Args:
        code: The source code to analyze
        
    Returns:
        Tuple of (is_safe, issues)
    """
    try:
        tree = ast.parse(code)
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        
        is_safe = len(analyzer.issues) == 0
        return is_safe, analyzer.issues
    except SyntaxError as e:
        return False, [f"Syntax error in code: {e}"]
    except Exception as e:
        return False, [f"Error analyzing code: {e}"]

def execute_submission_safely(submission_path, func_name="custom_kernel", *args, **kwargs):
    """
    Safely execute a user's submission by importing it in a sandboxed environment.
    
    Args:
        submission_path: Path to the submission.py file
        func_name: Name of the function to call in the submission
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        The result of the function call
    """
    # Get directory and filename
    dir_path = os.path.dirname(os.path.abspath(submission_path))
    file_name = os.path.basename(submission_path)
    module_name = os.path.splitext(file_name)[0]
    
    # Read the code and perform static analysis
    with open(submission_path, 'r') as f:
        code = f.read()
    
    is_safe, issues = analyze_code(code)
    if not is_safe:
        issues_str = "\n".join(issues)
        raise SecurityError(f"The submission contains potentially unsafe code:\n{issues_str}")
    
    # Save original directory and switch to submission directory
    original_dir = os.getcwd()
    original_path = sys.path.copy()
    
    try:
        os.chdir(dir_path)
        if dir_path not in sys.path:
            sys.path.insert(0, dir_path)
        
        # Use the sandbox for importing and executing
        with Sandbox():
            # Import the module
            module = importlib.import_module(module_name)
            
            # Get the function
            if not hasattr(module, func_name):
                raise AttributeError(f"Module {module_name} does not have a function named {func_name}")
            
            func = getattr(module, func_name)
            
            # Call the function
            return func(*args, **kwargs)
    finally:
        # Restore original directory and path
        os.chdir(original_dir)
        sys.path = original_path

class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass

class TimeoutError(Exception):
    """Raised when execution times out."""
    pass

class MemoryLimitError(Exception):
    """Raised when execution exceeds memory limits."""
    pass

@contextmanager
def time_limit(seconds):
    """
    Sets a time limit on code execution using SIGALRM.
    
    Args:
        seconds: Maximum execution time in seconds
    
    Raises:
        TimeoutError: If execution time exceeds the limit
    """
    # Define signal handler
    def signal_handler(signum, frame):
        raise TimeoutError(f"Code execution timed out after {seconds} seconds")
    
    # Set signal handler and alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Reset the alarm
        signal.alarm(0)

def set_memory_limit(memory_mb):
    """
    Sets a memory limit for the current process.
    
    Args:
        memory_mb: Maximum memory in megabytes
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    memory_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, hard))

def run_with_limitations(func, args=None, kwargs=None, timeout_sec=10, memory_mb=1024):
    """
    Runs a function with time and memory limitations.
    
    Args:
        func: The function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_sec: Maximum execution time in seconds
        memory_mb: Maximum memory usage in megabytes
        
    Returns:
        The result of the function call
    
    Raises:
        TimeoutError: If execution time exceeds the limit
        MemoryLimitError: If memory usage exceeds the limit
    """
    args = args or ()
    kwargs = kwargs or {}
    
    result = [None]
    exception = [None]
    
    def target():
        try:
            # Set memory limit for this thread
            set_memory_limit(memory_mb)
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    # Create and start thread
    thread = threading.Thread(target=target)
    
    try:
        with time_limit(timeout_sec):
            thread.start()
            thread.join(timeout=timeout_sec)
    except TimeoutError as e:
        # If we get here, the code timed out
        raise e
    
    if thread.is_alive():
        # Thread is still running after timeout
        raise TimeoutError(f"Code execution timed out after {timeout_sec} seconds")
    
    if exception[0]:
        # Re-raise any exception from the thread
        raise exception[0]
    
    return result[0]