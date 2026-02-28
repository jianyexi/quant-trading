import { useState, useEffect, useCallback } from 'react';
import { useTaskPoller } from './useTaskPoller';
import { cancelTask } from '../api/client';

/**
 * High-level task lifecycle hook.
 * Wraps useTaskPoller + sessionStorage persistence + output parsing + error handling.
 *
 * Usage:
 *   const tm = useTaskManager('task_gp');
 *   <button onClick={() => tm.submit(() => factorMineGP(opts))}>Run</button>
 *   <TaskOutput {...tm} />
 */
export function useTaskManager(storageKey: string) {
  const { task, startPolling, reset: resetPoller } = useTaskPoller();
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');

  const running = task?.status === 'Running' || task?.status === 'Pending';

  // Restore active task from sessionStorage on mount
  useEffect(() => {
    const savedId = sessionStorage.getItem(storageKey);
    if (savedId) startPolling(savedId);
  }, [storageKey, startPolling]);

  // Handle task completion/failure
  useEffect(() => {
    if (!task) return;
    if (task.status === 'Completed') {
      sessionStorage.removeItem(storageKey);
      try {
        const parsed = task.result ? JSON.parse(task.result) : null;
        let text = parsed?.stdout || task.result || '完成';
        if (parsed?.stderr) text += '\n\n--- stderr ---\n' + parsed.stderr;
        setOutput(text);
      } catch {
        setOutput(task.result || '完成');
      }
    } else if (task.status === 'Failed') {
      sessionStorage.removeItem(storageKey);
      setError(task.error || '任务失败');
    }
  }, [task, storageKey]);

  /** Submit a task. `apiFn` should return `{ task_id: string }`. */
  const submit = useCallback(async (apiFn: () => Promise<{ task_id: string }>) => {
    setError('');
    setOutput('');
    try {
      const result = await apiFn();
      sessionStorage.setItem(storageKey, result.task_id);
      startPolling(result.task_id);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '请求失败');
    }
  }, [storageKey, startPolling]);

  /** Cancel running task and clear state. */
  const cancel = useCallback(async () => {
    const taskId = sessionStorage.getItem(storageKey);
    if (taskId) {
      try { await cancelTask(taskId); } catch { /* ignore */ }
      sessionStorage.removeItem(storageKey);
    }
    resetPoller();
    setOutput('');
    setError('');
  }, [storageKey, resetPoller]);

  const progress = task?.progress || null;

  return { task, running, output, error, progress, submit, cancel, setOutput, setError };
}
